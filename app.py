#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards # beautify metric card with css
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import bcrypt
from datetime import datetime
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pymysql
from sqlalchemy import create_engine, text
import logging
import io
import time

# Configure logging
logging.basicConfig(filename='application.log', level=logging.INFO)

def connect_to_db():
    try:
        # Retrieve secrets from environment variables
        user = os.getenv('user')
        password = os.getenv('password')
        host = os.getenv('host')
        database = os.getenv('database')
        ssl_ca = os.getenv('ssl_ca')  # Ensure this is accessible in your environment

        connection_string = f'mysql+pymysql://{user}:{password}@{host}/{database}?ssl_ca={ssl_ca}'
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()

        return engine
    except Exception as err:
        st.sidebar.warning(f"Error: {err}")
        logging.error(f"Database connection error: {err}")
        return None
    
# Function to fetch latest data from the database
def fetch_latest_data(engine):
    try:
        with engine.connect() as connection:
            # Replace 'risk_data' with your actual table name
            result = connection.execute(text("SELECT * FROM risk_data ORDER BY date_last_updated DESC LIMIT 1"))
            latest_data = result.fetchone()
            return latest_data
    except Exception as e:
        st.error(f"An error occurred while fetching the latest data: {e}")
        return None

def fetch_risk_register_from_db():
    engine = connect_to_db()
    if engine:
        query = "SELECT * FROM risk_register"
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    return pd.DataFrame(columns=fetch_columns_from_risk_data())

def fetch_columns_from_risk_data():
    engine = connect_to_db()
    if engine:
        with engine.connect() as connection:
            result = connection.execute(text("DESCRIBE risk_data"))
            columns = [row[0] for row in result.fetchall()]
        engine.dispose()
        return columns
    return []

def insert_uploaded_data_to_db(dataframe):
    engine = connect_to_db()
    if engine:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                for _, row in dataframe.iterrows():
                    try:
                        date_last_updated = datetime.strptime(row['date_last_updated'], '%Y-%m-%d').date()
                    except ValueError:
                        date_last_updated = None
                    query = text("""
                        INSERT INTO risk_data (risk_description, risk_type, updated_by, date_last_updated, 
                                               cause_consequences, risk_owners, inherent_risk_probability, 
                                               inherent_risk_impact, inherent_risk_rating, control_owners, 
                                               residual_risk_probability, residual_risk_impact, 
                                               residual_risk_rating, controls) 
                        VALUES (:risk_description, :risk_type, :updated_by, :date_last_updated, :cause_consequences, 
                                :risk_owners, :inherent_risk_probability, :inherent_risk_impact, :inherent_risk_rating, 
                                :control_owners, :residual_risk_probability, :residual_risk_impact, :residual_risk_rating, 
                                :controls)
                    """)
                    connection.execute(query, row.to_dict())
                transaction.commit()
                st.sidebar.success("Data uploaded successfully!")
            except Exception as e:
                transaction.rollback()
                logging.error(f"Error inserting data: {e}")
                st.sidebar.error(f"Error inserting data: {e}")
        engine.dispose()
        
def insert_into_risk_data(data):
    engine = connect_to_db()
    if engine:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                # Ensure the @current_user_id session variable is set
                if 'user_id' in st.session_state:
                    user_id = st.session_state['user_id']
                    connection.execute(text("SET @current_user_id = :user_id"), {"user_id": user_id})
                else:
                    st.error("User ID not found in session state. Cannot proceed with insertion.")
                    return  # Early return if user_id is not set
                
                # Construct placeholders and columns from the data dictionary
                placeholders = ', '.join([f":{key}" for key in data.keys()])
                columns = ', '.join([f"`{key}`" for key in data.keys()])
                
                # Prepare the query using the text function
                query = text(f"INSERT INTO risk_data ({columns}) VALUES ({placeholders})")
                
                # Execute the query with the data dictionary
                connection.execute(query, data)  # Pass the data as a dictionary
                
                # Commit the transaction
                transaction.commit()
                logging.info(f"Inserted data: {data}")
            except Exception as e:
                # Rollback the transaction in case of error
                transaction.rollback()
                st.write(f"Error during insertion to risk_data: {e}")
                logging.error(f"Error during insertion: {e}")
            finally:
                # Ensure the connection is disposed in all cases
                engine.dispose()
   
      
# Function to fetch the latest data from the database
def fetch_latest_data(engine):
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM risk_data ORDER BY date_last_updated DESC"))
            return result.fetchall()
    except Exception as e:
        st.error(f"An error occurred while fetching the latest data: {e}")
        return None
    
# Fetch all data from the database
def fetch_all_from_risk_data(engine):
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM risk_data ORDER BY date_last_updated DESC"))
            return pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return pd.DataFrame()
    
def delete_from_risk_data_by_risk_description(risk_description):
    if 'user_role' in st.session_state and st.session_state.user_role in ['admin', 'superadmin']:
        engine = connect_to_db()
        if engine:
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    # Ensure the @current_user_id session variable is set
                    if 'user_id' in st.session_state:
                        user_id = st.session_state['user_id']
                        connection.execute(text("SET @current_user_id = :user_id"), {"user_id": user_id})
                    else:
                        st.error("User ID not found in session state. Cannot proceed with deletion.")
                        return

                    # Prepare and execute the delete statement
                    query = text("DELETE FROM risk_data WHERE TRIM(risk_description) = :risk_description")
                    result = connection.execute(query, {"risk_description": risk_description})
                    transaction.commit()

                    if result.rowcount > 0:
                        st.success(f"Risk '{risk_description}' deleted.")
                        logging.info(f"Deleted risk description: {risk_description}, Rows affected: {result.rowcount}")
                    else:
                        st.warning(f"No risk found with description '{risk_description}'.")
                except Exception as e:
                    transaction.rollback()
                    st.error(f"Error deleting risk: {e}")
                    logging.error(f"Error deleting risk {risk_description}: {e}")
            engine.dispose()
    else:
        st.error("You do not have permission to delete risks.")
        
def update_risk_data_by_risk_description(risk_description, updated_risk):
    # Check if the user has the required role to update the risk data
    if 'user_role' in st.session_state and st.session_state.user_role in ['admin', 'superadmin']:
        engine = connect_to_db()
        if not engine:
            st.sidebar.error("Database connection failed.")
            return

        with engine.connect() as connection:
            # Set the @current_user_id session variable
            user_id = st.session_state.user_id
            connection.execute(text("SET @current_user_id = :user_id"), {"user_id": user_id})

            # Prepare and execute the update statement
            update_query = text("""
            UPDATE risk_data
            SET
                risk_type = :risk_type,
                updated_by = :updated_by,
                date_last_updated = :date_last_updated,
                risk_description = :risk_description,
                cause_consequences = :cause_consequences,
                risk_owners = :risk_owners,
                inherent_risk_probability = :inherent_risk_probability,
                inherent_risk_impact = :inherent_risk_impact,
                inherent_risk_rating = :inherent_risk_rating,
                controls = :controls,
                Adequacy = :Adequacy,
                control_owners = :control_owners,
                residual_risk_probability = :residual_risk_probability,
                residual_risk_impact = :residual_risk_impact,
                residual_risk_rating = :residual_risk_rating,
                Direction = :Direction,
                Unit = :Unit,
                Status = :Status,
                opportunity_type = :opportunity_type
            WHERE
                risk_description = :risk_description_filter
            """)

            # Execute the update query with the updated risk data
            connection.execute(update_query, {
                "risk_type": updated_risk['risk_type'],
                "updated_by": updated_risk['updated_by'],
                "date_last_updated": updated_risk['date_last_updated'],
                "risk_description": updated_risk['risk_description'],
                "cause_consequences": updated_risk['cause_consequences'],
                "risk_owners": updated_risk['risk_owners'],
                "inherent_risk_probability": updated_risk['inherent_risk_probability'],
                "inherent_risk_impact": updated_risk['inherent_risk_impact'],
                "inherent_risk_rating": updated_risk['inherent_risk_rating'],
                "controls": updated_risk['controls'],
                "Adequacy": updated_risk['Adequacy'],  # Ensure the key matches what is used in the dictionary
                "control_owners": updated_risk['control_owners'],
                "residual_risk_probability": updated_risk['residual_risk_probability'],
                "residual_risk_impact": updated_risk['residual_risk_impact'],
                "residual_risk_rating": updated_risk['residual_risk_rating'],
                "Direction": updated_risk['Direction'],  # Ensure the key matches what is used in the dictionary
                "Unit": updated_risk['Unit'],
                "Status": updated_risk['Status'],
                "opportunity_type": updated_risk['opportunity_type'],
                "risk_description_filter": risk_description
            })

            # Notify the user that the risk has been updated successfully
            st.write("Risk updated successfully.")
    else:
        # If the user doesn't have the right permissions
        st.error("You do not have permission to update risks.")
      
def get_risk_id_by_description(risk_description):
    engine = connect_to_db()
    if engine:
        with engine.connect() as connection:
            query = text("SELECT id FROM risk_data WHERE TRIM(risk_description) = :risk_description")
            result = connection.execute(query, {"risk_description": risk_description})
            risk_id = result.fetchone()
        engine.dispose()
        return risk_id[0] if risk_id else None
    
def fetch_risks_outside_appetite_from_risk_data(risk_appetite):
    engine = connect_to_db()
    if engine:
        with engine.connect() as connection:
            # Use a parameterized query for a list of values
            placeholders = ', '.join([f":rating_{i}" for i in range(len(risk_appetite))])
            query = text(f"SELECT * FROM risk_data WHERE residual_risk_rating NOT IN ({placeholders})")
            # Create a dictionary with unique parameter names for each rating
            params = {f"rating_{i}": rating for i, rating in enumerate(risk_appetite)}
            result = connection.execute(query, params)
            data = pd.DataFrame(result.fetchall(), columns=result.keys())
        engine.dispose()
        return data
    return pd.DataFrame()

def insert_risks_into_risk_register(data):
    engine = connect_to_db()
    if engine:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                if isinstance(data, pd.DataFrame):
                    data_list = data.to_dict(orient='records')
                else:
                    data_list = [data]

                allowed_columns = ['risk_description', 'risk_type', 'updated_by', 'date_last_updated', 
                                   'cause_consequences', 'risk_owners', 'inherent_risk_probability', 
                                   'inherent_risk_impact', 'inherent_risk_rating', 'control_owners', 
                                   'residual_risk_probability', 'residual_risk_impact', 'residual_risk_rating', 
                                   'controls']
                
                for record in data_list:
                    record = {k: v for k, v in record.items() if k in allowed_columns}
                    
                    placeholders = ', '.join([f":{key}" for key in record.keys()])
                    columns = ', '.join(record.keys())
                    query = text(f"INSERT INTO risk_register ({columns}) VALUES ({placeholders})")
                    
                    logging.info(f"Executing query: {query} with parameters: {record}")
                    connection.execute(query, record)
                
                transaction.commit()
                logging.info(f"Inserted into risk_register: {data_list}")
            except Exception as e:
                transaction.rollback()
                logging.error(f"Error inserting into risk_register: {e}")
        engine.dispose()

def fetch_all_from_risk_register():
    engine = connect_to_db()
    if engine:
        query = "SELECT * FROM risk_register"
        data = pd.read_sql(query, engine)
        engine.dispose()
        return data
    return pd.DataFrame()

def update_risk_register_by_risk_description(risk_description, data):
    if 'user_role' in st.session_state and st.session_state.user_role == 'admin':
        engine = connect_to_db()
        if engine:
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    set_clause = ", ".join([f"{key} = :{key}" for key in data.keys()])
                    query = text(f"UPDATE risk_register SET {set_clause} WHERE risk_description = :risk_description")
                    connection.execute(query, data)
                    transaction.commit()
                    st.success("Risk updated successfully.")
                    logging.info(f"Updated risk_register for {risk_description}: {data}")
                except Exception as e:
                    transaction.rollback()
                    st.error(f"Error updating risk register: {e}")
                    logging.error(f"Error updating risk register {risk_description}: {e}")
            engine.dispose()
    else:
        st.error("You do not have permission to update risks.")

def delete_from_risk_register_by_risk_description(risk_description):
    if 'user_role' in st.session_state and st.session_state.user_role == 'admin':
        engine = connect_to_db()
        if engine:
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    query = text("DELETE FROM risk_register WHERE risk_description = :risk_description")
                    connection.execute(query, {"risk_description": risk_description})
                    transaction.commit()
                    st.success(f"Risk '{risk_description}' deleted.")
                    logging.info(f"Deleted risk_register description: {risk_description}")
                except Exception as e:
                    transaction.rollback()
                    st.error(f"Error deleting risk: {e}")
                    logging.error(f"Error deleting risk {risk_description}: {e}")
            engine.dispose()
    else:
        st.error("You do not have permission to delete risks.")
        
def register(user, password):
    engine = connect_to_db()
    if engine is None:
        logging.error("Failed to connect to the database.")
        return False
    
    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        logging.debug(f"Hashed password for {user}: {hashed_password}")
    except Exception as e:
        logging.error(f"Password hashing failed for {user}: {e}")
        st.sidebar.warning(f"Password hashing error: {e}")
        return False

    try:
        with engine.connect() as connection:
            query = text("INSERT INTO credentials (user, password) VALUES (:user, :password)")
            result = connection.execute(query, {"user": user, "password": hashed_password.decode('utf-8')})
            connection.commit()  # Ensure the transaction is committed
            logging.info(f"Registered new user {user}, Rows affected: {result.rowcount}")
        return True
    except Exception as err:
        logging.error(f"Registration error for user {user}: {err}")
        st.sidebar.warning(f"Error: {err}")
        return False
    
# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'user' not in st.session_state:
    st.session_state.user = ""

if 'user_role' not in st.session_state:
    st.session_state.user_role = ""
    
# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    
if 'user' not in st.session_state:
    st.session_state.user = ""

if 'user_role' not in st.session_state:
    st.session_state.user_role = ""
    
def login(user, password):
    logging.info(f"Attempting login for username: {user}")
    engine = connect_to_db()
    if engine:
        try:
            with engine.connect() as connection:
                # Adjusted query to include 'id' field
                query = text("SELECT id, password, expiry_date, role FROM credentials WHERE user = :user")
                result = connection.execute(query, {"user": user})
                row = result.fetchone()

                if row:
                    # Unpack the row to include 'id'
                    user_id, stored_password, expiry_date, role = row
                    logging.info(f"Fetched credentials for {user}")

                    if stored_password:
                        if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
                            logging.info(f"Password matched for {user}")

                            if expiry_date:
                                try:
                                    expiry_date = datetime.strptime(str(expiry_date), '%Y-%m-%d')
                                    if expiry_date < datetime.now():
                                        st.sidebar.error("Your account has expired. Please contact the administrator.")
                                        logging.info(f"Account expired for {user}")
                                        return False
                                except ValueError:
                                    st.sidebar.error("Invalid expiry date format. Please contact the administrator.")
                                    logging.error(f"Invalid expiry date format for {user}: {expiry_date}")
                                    return False

                            if not role:
                                st.sidebar.error("No role found for the user. Please contact the administrator.")
                                logging.error(f"No role found for {user}")
                                return False

                            # If login is successful
                            st.session_state.logged_in = True
                            st.session_state.user = user
                            st.session_state.user_role = role
                            st.session_state.user_id = user_id  # Store the user_id in session state
                            logging.info(f"User {user} logged in successfully with role {role} and ID {user_id}.")
                            return True
                        else:
                            logging.info(f"Invalid credentials for {user}")
                            return False
                    else:
                        logging.error(f"Stored password is missing for {user}")
                        return False
                else:
                    logging.info(f"Username not found: {user}")
                    return False
        except Exception as e:
            logging.error(f"Login error: {e}")
        finally:
            engine.dispose()
    return False
    
def logout():
    """Logout the user and clear session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.logged_in = False
    
def change_password(user, current_password, new_password):
    logging.info(f"Initiating password change for user: {user}")
    engine = connect_to_db()
    if engine:
        try:
            with engine.begin() as connection:  # Use a transaction
                # Verify the current password
                query = text("SELECT password FROM credentials WHERE user = :user")
                result = connection.execute(query, {"user": user})
                row = result.fetchone()
                
                if row:
                    stored_password = row[0]
                    logging.info(f"Stored password hash: {stored_password}")

                    # Check if the current password matches the stored password
                    if bcrypt.checkpw(current_password.encode('utf-8'), stored_password.encode('utf-8')):
                        logging.info("Current password verified successfully.")

                        # Hash the new password
                        new_hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                        logging.info(f"New hashed password: {new_hashed_password}")

                        # Update with the new password
                        update_query = text("UPDATE credentials SET password = :new_password WHERE user = :user")
                        result = connection.execute(update_query, {"new_password": new_hashed_password, "user": user})
                        if result.rowcount == 1:
                            logging.info("Password updated in the database.")
                            return True
                        else:
                            logging.error("Password update failed, no rows affected.")
                            st.sidebar.error("Password update failed.")
                            return False
                    else:
                        logging.warning("Current password verification failed.")
                        st.sidebar.error("The current password you entered is incorrect.")
                        return False
                else:
                    logging.warning(f"User {user} not found in the database.")
                    st.sidebar.error("User not found.")
                    return False
        except Exception as e:
            logging.error(f"Error during password change: {e}")
            st.sidebar.error("An error occurred while changing the password.")
            return False
    else:
        logging.error("Failed to connect to the database.")
        st.sidebar.error("Could not connect to the database.")
        return False
    
# Define the risk appetite based on risk type
def get_risk_appetite(risk_type):
    risk_appetite_map = {
        'Strategic Risk': ['Moderate', 'High', 'Critical'],
        'Operational Risk': ['Low', 'Moderate', 'High'],
        'Compliance Risk': ['Low', 'Moderate', 'High'],
        'Reputational Risk': ['Low', 'Moderate', 'High'],
        'Financial Risk': ['Low', 'Moderate', 'High'],
        'Regulatory Risk': ['Moderate', 'High', 'Critical'],
        'Enviromental Risk': ['Low', 'Moderate'],
        'Human Resource Risk': ['Moderate', 'High'],
        'Supply Chain Risk': ['Low', 'Moderate', 'High'],
        'Ethical Risk': ['Low', 'Moderate'],
        'Technlogical Risk': ['Moderate', 'High', 'Critical'],
        'Public Health Risk': ['Low', 'Moderate', 'High']
    }
    return risk_appetite_map.get(risk_type, [])

# Define unit and add 'All' option, then sort alphabetically
units = [
    'All',
    'Licensing and Enforcement',
    'Evaluations and Registration',
    'Pharmacovigilance and Clinical Trials',
    'Chemistry Laboratory',
    'Microbiology Laboratory',
    'Medical Devices Laboratory',
    'Quality Unit',
    'Legal Unit',
    'Human Resources',
    'Information and Communication Technology',
    'Finance and Administration'                            
]
units.sort()

# Use unit in any part of the application

# Function to get risk by description
def fetch_risk_by_description(risk_description):
    # Assuming you're using SQLAlchemy
    engine = connect_to_db()
    connection = engine.connect()
    
    query = text("""
    SELECT * FROM risk_data 
    WHERE risk_description = :description
    LIMIT 1;
    """)
    
    result = connection.execute(query, {"description": risk_description}).fetchone()
    connection.close()
    
    if result:
        return dict(result._mapping)
    else:
        return None
    
def make_autopct(counts):
    def my_autopct(pct):
        total = sum(counts)
        val = int(round(pct * total / 100.0))
        return f'{pct:.1f}%\n({val})'
    return my_autopct
    
def fetch_residual_risk_rating_distribution():
    engine = connect_to_db()
    query = "SELECT residual_risk_rating, COUNT(*) as count FROM risk_data GROUP BY residual_risk_rating"
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    return df

# Function to filter and generate trend analysis
def generate_trend_analysis(risk_data):
    # Convert date_last_updated to datetime
    risk_data['date_last_updated'] = pd.to_datetime(risk_data['date_last_updated'])
    
    # Filters
    risk_types = st.multiselect('Select Risk Type(s)', risk_data['risk_type'].unique(), default=risk_data['risk_type'].unique())
    risk_owners = st.multiselect('Select Risk Owner(s)', risk_data['risk_owners'].unique(), default=risk_data['risk_owners'].unique())
    
    # Date filters with "From" and "To"
    date_from = st.date_input("From Date", value=risk_data['date_last_updated'].min())
    date_to = st.date_input("To Date", value=risk_data['date_last_updated'].max())
    
    # Ensure that 'date_from' is before 'date_to'
    if date_from > date_to:
        st.error("'From Date' must be earlier than 'To Date'. Please correct the dates.")
        return

    # Filter data
    filtered_data = risk_data[
        (risk_data['risk_type'].isin(risk_types)) &
        (risk_data['risk_owners'].isin(risk_owners)) &
        (risk_data['date_last_updated'] >= pd.to_datetime(date_from)) &
        (risk_data['date_last_updated'] <= pd.to_datetime(date_to))
    ]
    
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
    else:
        # Trend analysis by risk type
        trend_data = filtered_data.groupby(['date_last_updated', 'risk_type']).size().unstack(fill_value=0)
        
        st.write("Trend Analysis by Risk Type")
        st.line_chart(trend_data)
        
        # Show filtered data
        st.write("Filtered Data", filtered_data)
        
        # Allow download of the filtered data
        csv = filtered_data.to_csv(index=False)
        st.download_button(label="Download Filtered Data as CSV", data=csv, file_name="filtered_risk_data.csv", mime="text/csv")
        
# Function to calculate KPIs Critical
def calculate_kpis(df):
    # 1. Risk Reduction KPI
    initial_critical_risks = df[df['inherent_risk_rating'] == 'Critical'].shape[0]
    reduced_critical_risks = df[(df['inherent_risk_rating'] == 'Critical') & (df['residual_risk_rating'] != 'Critical')].shape[0]
    risk_reduction_kpi = (reduced_critical_risks / initial_critical_risks) * 100 if initial_critical_risks > 0 else 0

    # 2. Action Completion KPI (Assuming 'Status' indicates completion)
    total_actions = df.shape[0]
    completed_actions = df[df['Status'] == 'Closed'].shape[0]
    action_completion_kpi = (completed_actions / total_actions) * 100 if total_actions > 0 else 0

    # 3. Cost Performance KPI (If cost data were available, this would compare actual vs. budgeted costs)

    # 4. Residual Risk KPI (Percentage of risks that are still rated 'Critical')
    residual_critical_risks = df[df['residual_risk_rating'] == 'Critical'].shape[0]
    residual_risk_kpi = (residual_critical_risks / total_actions) * 100 if total_actions > 0 else 0

    return {
        "Risk Reduction KPI (%)": risk_reduction_kpi,
        "Action Completion KPI (%)": action_completion_kpi,
        "Residual Risk KPI (%)": residual_risk_kpi
    }

# Function to calculate KPIs High
def calculate_kpis_high(df):
    # 1. Risk Reduction KPI
    initial_high_risks = df[df['inherent_risk_rating'] == 'High'].shape[0]
    reduced_high_risks = df[(df['inherent_risk_rating'] == 'High') & (df['residual_risk_rating'] != 'High')].shape[0]
    risk_reduction_kpi = (reduced_high_risks / initial_high_risks) * 100 if initial_high_risks > 0 else 0

    # 2. Action Completion KPI (Assuming 'Status' indicates completion)
    total_actions = df.shape[0]
    completed_actions = df[df['Status'] == 'Closed'].shape[0]
    action_completion_kpi = (completed_actions / total_actions) * 100 if total_actions > 0 else 0

    # 3. Cost Performance KPI (If cost data were available, this would compare actual vs. budgeted costs)

    # 4. Residual Risk KPI (Percentage of risks that are still rated 'Critical')
    residual_high_risks = df[df['residual_risk_rating'] == 'High'].shape[0]
    residual_risk_kpi = (residual_high_risks / total_actions) * 100 if total_actions > 0 else 0

    return {
        "Risk Reduction KPI (%)": risk_reduction_kpi,
        "Action Completion KPI (%)": action_completion_kpi,
        "Residual Risk KPI (%)": residual_risk_kpi
    }

# Function to generate progress reports
def generate_progress_reports(df):
    # Generate report summary
    report_summary = df.groupby(['Status']).size().reset_index(name='Count')
    return report_summary

# Function to create a risk dashboard
def create_risk_dashboard(df):
    # Create a dashboard with a summary of key metrics
    risk_type_summary = df.groupby(['risk_type', 'residual_risk_rating']).size().reset_index(name='Count')
    return risk_type_summary        

def main():
    st.image("logo.png", width=400)
    st.markdown('### Enterprise Risk Assessment Application')
     
    if not st.session_state.logged_in:
        st.sidebar.header("Login")
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login", key="login_button"):
            if login(username, password):
                st.sidebar.success(f"Logged in as: {st.session_state.user}")
                st.sidebar.info(f"Role: {st.session_state.user_role}")
            else:
                st.sidebar.error("Login failed. Please check your credentials.")
    else:
        st.sidebar.header(f"Welcome, {st.session_state.user}")
        st.sidebar.info(f"Role: {st.session_state.user_role}")
        if st.sidebar.button("Logout", key="logout_button"):
            logout()
            st.sidebar.success("Logged out successfully!")

        # Password change form in the sidebar
        st.sidebar.subheader("Change Password")
        current_password = st.sidebar.text_input("Current Password", type="password", key="current_password")
        new_password = st.sidebar.text_input("New Password", type="password", key="new_password")
        confirm_new_password = st.sidebar.text_input("Confirm New Password", type="password", key="confirm_new_password")
        if st.sidebar.button("Change Password"):
            if new_password == confirm_new_password:
                if change_password(st.session_state.user, current_password, new_password):
                    st.sidebar.success("Password changed successfully.")
                else:
                    st.sidebar.error("Current password is incorrect.")
            else:
                st.sidebar.error("New passwords do not match.")

    # Additional application logic goes here
    if st.session_state.logged_in:
        st.write(f"Welcome {st.session_state.user}! You are logged in as {st.session_state.user_role} and User ID {st.session_state.user_id}.")
    else:
        st.write("Please log in to access the application.")
    
    if st.session_state.logged_in and st.session_state.user_role == 'superadmin':
        st.sidebar.subheader("Register New User")
        new_username = st.sidebar.text_input("New Username", key='reg_username')
        new_password = st.sidebar.text_input("New Password", type="password", key='reg_password')
        if st.sidebar.button("Register"):
            if register(new_username, new_password):
                st.sidebar.success("Registered successfully! The new user can now log in.")
            else:
                st.sidebar.error("Registration failed. The username might already be taken.")
    elif st.session_state.logged_in:
        st.sidebar.info("Only super admin users can register new users.")

    if st.session_state.logged_in:
        # Main application content goes here
   
        def plot_risk_matrix():
            fig = plt.figure(figsize=(10, 10))
            plt.subplots_adjust(wspace=0, hspace=0)
            
            # Setting y-ticks with corresponding percentages
            plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], 
                       ['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5], 
                       ['Very Low\n(0%-10%)', 'Low\n(11%-25%)', 'Medium\n(26%-50%)', 'High\n(51%-90%)', 'Very High\n(91%-100%)'])

            plt.xlim(0, 5)
            plt.ylim(0, 5)
            plt.xlabel('Impact', fontsize=18)
            plt.ylabel('Probability', fontsize=18)

            nrows = 5
            ncols = 5
            axes = [fig.add_subplot(nrows, ncols, r * ncols + c + 1) for r in range(nrows) for c in range(ncols)]

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            # Assigning colors to the matrix cells
            green = [10, 15, 16, 20, 21, 22, 5]
            yellow = [0, 6, 17, 23, 11, 12]
            orange = [1, 2, 7, 13, 18, 24]
            red = [3, 4, 8, 9, 14, 19]

            for index in green:
                axes[index].set_facecolor('green')
            for index in yellow:
                axes[index].set_facecolor('yellow')
            for index in orange:
                axes[index].set_facecolor('orange')
            for index in red:
                axes[index].set_facecolor('red')

            # Adding text labels to the matrix cells
            labels = {
                'Low': green,
                'Moderate': yellow,
                'High': orange,
                'Critical': red
            }

            for label, positions in labels.items():
                for pos in positions:
                    axes[pos].text(0.5, 0.5, label, ha='center', va='center', fontsize=14)

            return fig  # Return the figure object

       
        risk_levels = {
            'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5
        }
        
        risk_rating_dict = {
            (1, 1): 'Low', (1, 2): 'Low', (1, 3): 'Low', (1, 4): 'Low', (1, 5): 'Moderate',
            (2, 1): 'Low', (2, 2): 'Low', (2, 3): 'Moderate', (2, 4): 'Moderate', (2, 5): 'High',
            (3, 1): 'Low', (3, 2): 'Moderate', (3, 3): 'Moderate', (3, 4): 'High', (3, 5): 'High',
            (4, 1): 'Moderate', (4, 2): 'High', (4, 3): 'High', (4, 4): 'Critical', (4, 5): 'Critical',
            (5, 1): 'High', (5, 2): 'Critical', (5, 3): 'Critical', (5, 4): 'Critical', (5, 5): 'Critical'
        }

      
       
#         def calculate_risk_rating(probability, impact):
#             risk_level_num = risk_levels.get(probability, None), risk_levels.get(impact, None)
#             rating = risk_rating_dict.get(risk_level_num, 'Unknown')
#             if rating == 'Low':
#                 rating = 'Medium'  # Correcting the erroneous 'Low' rating
#             return rating


        def calculate_risk_rating(probability, impact):
            return risk_rating_dict[(risk_levels[probability], risk_levels[impact])]

        tab = st.sidebar.selectbox(
            'Choose a function',
            ('Risk Matrix', 'Main Application', 'Risks Overview', 'Risks Owners & Control Owners', 
             'Adjusted Risk Matrices', 'Performance Metrics', 'Reports','Delete Risk', 'Update Risk', 
             'Delete/Update User')
        )

        if 'risk_data' not in st.session_state:
            
            engine = connect_to_db()

            st.session_state['risk_data'] = fetch_all_from_risk_data(engine)
            if st.session_state['risk_data'].empty:
                st.session_state['risk_data'] = pd.DataFrame(columns=[
                    'risk_description', 'cause_consequences', 'risk_owners', 
                    'inherent_risk_probability', 'inherent_risk_impact', 'inherent_risk_rating',
                    'controls', 'control_owners', 
                    'residual_risk_probability', 'residual_risk_impact', 'residual_risk_rating'
                ])

        if 'risk_register' not in st.session_state:
            st.session_state['risk_register'] = fetch_risk_register_from_db()

        if 'risk_type' not in st.session_state:
            st.session_state['risk_type'] = ''
            
        if 'risk_appetite' not in st.session_state:
            st.session_state['risk_appetite'] = ''

        if 'updated_by' not in st.session_state:
            st.session_state['updated_by'] = ''

        if 'date_last_updated' not in st.session_state:
            st.session_state['date_last_updated'] = pd.to_datetime('today')

        if tab == 'Risk Matrix':
            if 'risk_data' not in st.session_state:
                st.session_state['risk_data'] = fetch_all_from_risk_data()

            st.subheader('Master Risk Matrix')

            # Call the function to create the figure
            fig = plot_risk_matrix()

            if fig:
                # Display the figure in the Streamlit app
                st.pyplot(fig)

                # Get the current datetime and format it
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Construct the file name with the current datetime
                file_name = f"risk_matrix_{current_datetime}.png"

                # Save the figure to a BytesIO object
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)

                # Download button
                st.download_button(
                    label="Download Risk Matrix as PNG",
                    data=buf,
                    file_name=file_name,
                    mime="image/png"
                )
            else:
                st.error("Error in creating the risk matrix plot.")
                
            st.subheader('Risk Appetite Matrix')
            
            # Define risk types and their corresponding appetite levels
            risk_data = [
                ("Strategic Risk", "Moderate", "High", "Critical"),
                ("Operational Risk", "Low", "Moderate", "High"),
                ("Compliance Risk", "Low", "Moderate", "High"),
                ("Reputation Risk", "Low", "Moderate", "High"),
                ("Financial Risk", "Low", "Moderate", "High"),
                ("Regulatory Risk", "Moderate", "High", "Critical"),
                ("Environmental Risk", "Low", "Moderate"),
                ("Human Resources Risk", "Moderate", "High"),
                ("Supply Chain Risk", "Low", "Moderate", "High"),
                ("Ethical Risk", "Low", "Moderate"),
                ("Technologica Risk", "Moderate", "High", "Critical"),
                ("Public Health Risk", "Low", "Moderate", "High")
            ]
            
                       
            # Sort the risks alphabetically by the first element in each tuple
            risk_data.sort(key=lambda x: x[0])

            # Define colors for risk levels
            color_map = {
                'High': 'orange',
                'Critical': 'red',
                'Low': 'green',
                'Moderate': 'yellow'
            }
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot rectangles for each risk type
            for i, risk in enumerate(risk_data):
                risk_type = risk[0]
                appetites = risk[1:]  # All appetite levels after the risk type
                num_levels = len(appetites)

                # Determine the width of each rectangle based on the number of levels
                rect_width = 1.0 / num_levels

                for j, appetite in enumerate(appetites):
                    ax.add_patch(plt.Rectangle((j * rect_width, i), rect_width, 1, facecolor=color_map[appetite], edgecolor='black'))
                    ax.text((j + 0.5) * rect_width, i + 0.5, appetite, ha='center', va='center', fontsize=16)

            # Set y-axis ticks and labels
            ax.set_yticks(np.arange(len(risk_data)) + 0.5)
            ax.set_yticklabels([risk[0] for risk in risk_data], va='center', ha='right', rotation=0, fontsize=18)
            ax.tick_params(axis='y', which='major', pad=10)  # Add padding to y-axis labels

            # Remove x-axis ticks
            ax.set_xticks([])

            # Set title
            ax.set_title('Risk Appetite Matrix', pad=20, fontsize=24)

            # Remove axes
            ax.spines[:].set_visible(False)

            # Set limits to show full rectangles
            ax.set_xlim(0, 1)
            ax.set_ylim(0, len(risk_data))

            # Adjust layout and display
            plt.tight_layout()
            plt.ylabel('Risks', fontsize=18)

            if fig:
                # Display the figure in the Streamlit app
                st.pyplot(fig)

                # Get the current datetime and format it
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Construct the file name with the current datetime
                file_name = f"risk_appetite_matrix_{current_datetime}.png"

                # Save the figure to a BytesIO object
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)

                # Download button
                st.download_button(
                    label="Download Risk Appetite Matrix as PNG",
                    data=buf,
                    file_name=file_name,
                    mime="image/png"
                )
            else:
                st.error("Error in creating the risk appetite matrix plot.")
        
        elif tab == 'Main Application':
            
            engine = connect_to_db()
            
            if 'risk_data' not in st.session_state:
                st.session_state['risk_data'] = fetch_all_from_risk_data(engine) 
                                 
            st.subheader('Enter Risk Details')
          
            st.session_state['risk_type'] = st.selectbox('Risk Type', sorted([
                'Strategic Risk', 'Operational Risk', 'Compliance Risk', 'Reputational Risk', 'Financial Risk',
                'Regulatory Risk', 'Envioronmental Risk', 'Human Resource Risk',
                'Supply Chain Risk', 'Ethical Risk', 'Technological Risk', 'Public Health Risk'
            ]))
            
            st.session_state['updated_by'] = st.text_input('Updated By')
            st.session_state['date_last_updated'] = st.date_input('Date Last Updated')
            risk_description = st.text_input('Risk Description', key='risk_description')
            cause_consequences = st.text_input('Cause & Consequences', key='cause_consequences')
            risk_owners = st.text_input('Risk Owner(s)', key='risk_owners')
            inherent_risk_probability = st.selectbox('Inherent Risk Probability', list(risk_levels.keys()), key='inherent_risk_probability')
            inherent_risk_impact = st.selectbox('Inherent Risk Impact', list(risk_levels.keys()), key='inherent_risk_impact')
            controls = st.text_input('Control(s)', key='controls')

            # New field for Adequacy
            adequacy = st.selectbox('Adequacy', ['Weak', 'Acceptable', 'Strong'], key='adequacy')

            control_owners = st.text_input('Control Owner(s)', key='control_owners')
            residual_risk_probability = st.selectbox('Residual Risk Probability', list(risk_levels.keys()), key='residual_risk_probability')
            residual_risk_impact = st.selectbox('Residual Risk Impact', list(risk_levels.keys()), key='residual_risk_impact')

            # New field for Direction
            direction = st.selectbox('Direction', ['Increasing', 'Decreasing', 'Stable'], key='direction')

            # New field for Status
            status = st.selectbox('Status', ['Open', 'Closed'], key='status')

            # New field for unit
            st.session_state['unit'] = st.selectbox('Unit', sorted([
                'Licensing and Enforcement', 'Evaluations and Registration', 'Pharmacovigilance and Clinical Trials',
                'Chemistry Laboratory', 'Microbiology Laboratory', 'Medical Devices Laboratory', 'Quality Unit',
                'Legal Unit', 'Human Resources', 'Information and Communication Technology', 'Finance and Administration'
            ]))

            # New field for Opportunity Type
            opportunity_type = st.selectbox('Is there an Opportunity associated with this risk?', ['No', 'Yes'], key='opportunity_type')

            if st.button('Enter Risk'):
                inherent_risk_rating = calculate_risk_rating(inherent_risk_probability, inherent_risk_impact)
                residual_risk_rating = calculate_risk_rating(residual_risk_probability, residual_risk_impact)

                new_risk = {
                    'risk_type': st.session_state['risk_type'],
                    'updated_by': st.session_state['updated_by'],
                    'date_last_updated': st.session_state['date_last_updated'],
                    'risk_description': risk_description,
                    'cause_consequences': cause_consequences,
                    'risk_owners': risk_owners, 
                    'inherent_risk_probability': inherent_risk_probability,
                    'inherent_risk_impact': inherent_risk_impact,
                    'inherent_risk_rating': inherent_risk_rating,
                    'controls': controls,
                    'adequacy': adequacy,  # Include the new adequacy field
                    'control_owners': control_owners,
                    'residual_risk_probability': residual_risk_probability,
                    'residual_risk_impact': residual_risk_impact,
                    'residual_risk_rating': residual_risk_rating,
                    'direction': direction,  # Include the new direction field
                    'unit': st.session_state['unit'],  # Include the new unit field
                    'Status': status,  # Include the status field
                    'opportunity_type': opportunity_type  # Include the opportunity type field
                }
          
                # Check if a record with the same risk_description already exists
                existing_risk = fetch_risk_by_description(risk_description)

                if existing_risk:
                    st.warning(f"A risk with the description '{risk_description}' already exists. Please use a different description or update the existing risk.")
                else:
                    try:
                        insert_into_risk_data(new_risk)
                        st.success("New risk data successfully entered")

                        # Fetch and display the latest data after insertion
                        risk_data = fetch_all_from_risk_data(engine)  # Fetch fresh data
                        st.session_state['risk_data'] = risk_data  # Update session state with the latest data

                    except Exception as e:
                        st.error(f"Error inserting into risk_data: {e}")
                    
            st.subheader('Risk Filters')
        
            engine = connect_to_db()

            # Load or fetch data
            risk_data = st.session_state.get('risk_data', fetch_all_from_risk_data(engine))

            # Initialize filtered_data as an empty DataFrame
            filtered_data = pd.DataFrame()
            
            # Define colors for each risk rating
            colors = {
                'Low': 'background-color: green',
                'Moderate': 'background-color: yellow',
                'High': 'background-color: orange',
                'Critical': 'background-color: red'
            }
            
            # Define colors for each adequacy
            colors_adequacy = {
                'Weak': 'background-color: orange',
                'Acceptable': 'background-color: yellow',
                'Strong': 'background-color: green'
            }
            
            # Define colors for each direction
            colors_direction = {
                'Increasing': 'background-color: orange',
                'Decreasing': 'background-color: yellow',
                'Stable': 'background-color: green'
            }

            # Function to apply styles
            def highlight_risk(rating):
                return colors.get(rating, '')
            
            # Function to highlight adequacy based on value
            def highlight_adequacy(val):
                return colors_adequacy.get(val, '')

            # Function to highlight direction based on value
            def highlight_direction(val):
                return colors_direction.get(val, '')

            # Check if the DataFrame is not empty and contains the 'date_last_updated' column
            if not risk_data.empty and 'date_last_updated' in risk_data.columns:
                # Ensure 'date_last_updated' is in datetime format, coerce errors to NaT
                risk_data['date_last_updated'] = pd.to_datetime(risk_data['date_last_updated'], errors='coerce')

                # Date filter section: ensure min_date and max_date are valid
                min_date = risk_data['date_last_updated'].min()
                max_date = risk_data['date_last_updated'].max()

                # Handle cases where min_date or max_date might be NaT
                if pd.isnull(min_date):
                    min_date = datetime.today().date()  # Set to today's date if NaT
                else:
                    min_date = min_date.date()  # Convert to datetime.date

                if pd.isnull(max_date):
                    max_date = datetime.today().date()  # Set to today's date if NaT
                else:
                    max_date = max_date.date()  # Convert to datetime.date

                # Use the dates in the Streamlit date input, ensuring they are valid
                from_date = st.date_input('From', value=min_date, min_value=min_date, max_value=max_date)
                to_date = st.date_input('To', value=max_date, min_value=min_date, max_value=max_date)

                # Apply date filter to the data, ensuring proper conversion to Timestamp
                filtered_data = risk_data[(risk_data['date_last_updated'] >= pd.Timestamp(from_date)) &
                                          (risk_data['date_last_updated'] <= pd.Timestamp(to_date))]
            else:
                st.warning("The data is empty or missing the 'date_last_updated' column.")
                     
            # Add a selectbox for unit filtering
            selected_unit = st.selectbox('Select Unit', units)

            # Apply unit filter if not 'All'
            if selected_unit != 'All':
                filtered_data = filtered_data[(filtered_data['Unit'] == selected_unit) | (filtered_data['Unit'].isna())]
                
            # Add a filter for risk owners
            risk_owners = ['All'] + sorted(risk_data['risk_owners'].dropna().unique().tolist())
            selected_risk_owner = st.selectbox('Select Risk Owner', risk_owners)

            # Apply risk owners filter if not 'All'
            if selected_risk_owner != 'All':
                filtered_data = filtered_data[filtered_data['risk_owners'] == selected_risk_owner] 
                
            # Initialize filtered_data
            filtered_data = risk_data.copy() if not risk_data.empty else pd.DataFrame()

            # Check if 'risk_category' exists and has data
            if 'risk_category' in risk_data.columns and not risk_data['risk_category'].dropna().empty:
                # Add a filter for risk category
                risk_categories = ['All'] + sorted(risk_data['risk_category'].dropna().unique().tolist())
            else:
                # Default to 'All' if no data or column missing
                risk_categories = ['All']

            # Selectbox for risk category
            selected_risk_category = st.selectbox('Select Risk Category', risk_categories)

            # Apply risk category filter if not 'All' and if column exists
            if selected_risk_category != 'All' and 'risk_category' in risk_data.columns:
                filtered_data = filtered_data[filtered_data['risk_category'] == selected_risk_category]
            else:
                st.warning("No risk category data available or no filter applied.")

            st.subheader('Risk Data')

            # Display the filtered data or a message if it's empty
            if filtered_data.empty:
                st.info("No data available for the selected date range and unit.")
            else:
                # Apply the style to all relevant columns
                styled_risk_data = filtered_data.style.applymap(highlight_risk, subset=['inherent_risk_rating', 'residual_risk_rating']) \
                    .applymap(highlight_adequacy, subset=['Adequacy']) \
                    .applymap(highlight_direction, subset=['Direction'])
#                 styled_risk_data = filtered_data.style.applymap(highlight_risk, subset=['inherent_risk_rating', 'residual_risk_rating'])

                # Display the styled dataframe in Streamlit
                st.dataframe(styled_risk_data)

                # Prepare data for download (unstyled data)
                csv = filtered_data.to_csv(index=False)
                current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')

                # Provide a download button for the CSV file
                st.download_button(
                    label="Download Risk Data",
                    data=csv,
                    file_name=f"risk_data_{current_datetime}.csv",
                    mime="text/csv",
                )

            st.subheader('Risk Register')
            
            # Check for required columns before applying further filtering
            required_columns = ['inherent_risk_rating', 'residual_risk_rating', 'risk_type', 'Adequacy', 'Direction', 'opportunity_type']

            if all(column in filtered_data.columns for column in required_columns):
                filtered_data['risk_appetite'] = filtered_data['risk_type'].apply(get_risk_appetite)

                def residual_exceeds_appetite(row):
                    # Define a mapping of risk levels for comparison purposes
                    risk_levels = ['Low', 'Moderate', 'High', 'Critical']

                    # Check if risk appetite is empty
                    if not row['risk_appetite']:
                        return False  # or True if you want to keep risks with no defined appetite

                    # Find the maximum level in the appetite for comparison
                    max_appetite_level = max(row['risk_appetite'], key=lambda level: risk_levels.index(level))

                    # Check if residual risk rating exceeds the maximum appetite level
                    exceeds_appetite = risk_levels.index(row['residual_risk_rating']) > risk_levels.index(max_appetite_level)

                    # Check if the risk is flagged as an opportunity
                    accepted_due_to_opportunity = row['opportunity_type'] == 'Yes'

                    if exceeds_appetite and not accepted_due_to_opportunity:
                        return True

                    return False

                # Filter the DataFrame based on the residual_exceeds_appetite function
                risk_register = filtered_data[filtered_data.apply(residual_exceeds_appetite, axis=1)]

                if not risk_register.empty:
                    # Apply the style to both columns
                    styled_risk_data = risk_register.style.applymap(highlight_risk, subset=['inherent_risk_rating', 'residual_risk_rating']) \
                        .applymap(highlight_adequacy, subset=['Adequacy']) \
                        .applymap(highlight_direction, subset=['Direction'])

                    # Display the styled DataFrame in Streamlit
                    st.dataframe(styled_risk_data)

                    # Convert the risk register to CSV format
                    csv_register = risk_register.to_csv(index=False)

                    # Generate a timestamp for the filename
                    current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')

                    # Provide a download button for the CSV file
                    st.download_button(
                        label="Download Risk Register",
                        data=csv_register,
                        file_name=f"risk_register_{current_datetime}.csv",
                        mime="text/csv",
                    )
                else:
                    st.write("No risk register data available to display or download.")
            else:
                st.warning("The data is missing required columns for filtering.")
        
            st.subheader('Opportunities Data')

            # Display the filtered data or a message if it's empty
            if filtered_data.empty:
                st.info("No data available for the selected date range and unit.")
            else:
                # Filter the data to show only opportunities
                opportunity_data = filtered_data[filtered_data['opportunity_type'] == 'Yes']

                # Check if there are any opportunities after filtering
                if opportunity_data.empty:
                    st.info("No opportunities available.")
                else:
                    # Apply the style to all relevant columns
                    styled_risk_data = opportunity_data.style.applymap(highlight_risk, subset=['inherent_risk_rating', 'residual_risk_rating']) \
                        .applymap(highlight_adequacy, subset=['Adequacy']) \
                        .applymap(highlight_direction, subset=['Direction'])

                    # Display the styled dataframe in Streamlit
                    st.dataframe(styled_risk_data)

                    # Prepare data for download (unstyled data)
                    csv = opportunity_data.to_csv(index=False)
                    current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')

                    # Provide a download button for the CSV file
                    st.download_button(
                        label="Download Opportunities Data",
                        data=csv,
                        file_name=f"opportunity_data_{current_datetime}.csv",
                        mime="text/csv",
                    )
            
        elif tab == 'Risks Overview':
            st.markdown("""
            <style>
                body .stMetric span:first-child {
                    font-size: 12px !important; 
                }
                body .stMetric span:last-child {
                    font-size: 16px !important;
                }
            </style>
            """, unsafe_allow_html=True)

            if 'risk_data' not in st.session_state:
                st.session_state['risk_data'] = fetch_all_from_risk_data()
                
            # Streamlit layout
            st.header('Risks Dashboard')
            st.subheader('Risk Filters')
            
            engine = connect_to_db()
            
            # Load data from session state
            risk_data = st.session_state.get('risk_data', fetch_all_from_risk_data(engine))

            # Initialize filtered_data as an empty DataFrame
            filtered_data = pd.DataFrame()

            # Check if the DataFrame is not empty and contains the 'date_last_updated' column
            if not risk_data.empty and 'date_last_updated' in risk_data.columns:
                # Ensure 'date_last_updated' is in datetime format, coerce errors to NaT
                risk_data['date_last_updated'] = pd.to_datetime(risk_data['date_last_updated'], errors='coerce')

                # Date filter section: ensure min_date and max_date are valid
                min_date = risk_data['date_last_updated'].min()
                max_date = risk_data['date_last_updated'].max()

                # Handle cases where min_date or max_date might be NaT
                if pd.isnull(min_date):
                    min_date = datetime.today().date()
                else:
                    min_date = min_date.date()

                if pd.isnull(max_date):
                    max_date = datetime.today().date()
                else:
                    max_date = max_date.date()

                # Use the dates in the Streamlit date input
                from_date = st.date_input('From', value=min_date, min_value=min_date, max_value=max_date)
                to_date = st.date_input('To', value=max_date, min_value=min_date, max_value=max_date)

                # Apply date filter to the data
                filtered_data = risk_data[(risk_data['date_last_updated'] >= pd.Timestamp(from_date)) &
                                          (risk_data['date_last_updated'] <= pd.Timestamp(to_date))]
            else:
                st.warning("The data is empty or missing the 'date_last_updated' column.")

            # Add a selectbox for unit filtering
            selected_unit = st.selectbox('Select Unit', units)

            # Apply unit filter if not 'All'
            if selected_unit != 'All':
                filtered_data = filtered_data[(filtered_data['Unit'] == selected_unit) | (filtered_data['Unit'].isna())]
                
            # Add a filter for risk owners
            risk_owners = ['All'] + sorted(risk_data['risk_owners'].dropna().unique().tolist())
            selected_risk_owner = st.selectbox('Select Risk Owner', risk_owners)

            # Apply risk owners filter if not 'All'
            if selected_risk_owner != 'All':
                filtered_data = filtered_data[filtered_data['risk_owners'] == selected_risk_owner]

            st.subheader('Risk Data')

            # Display the filtered data or a message if it's empty
            if filtered_data.empty:
                st.info("No data available for the selected date range and unit.")
            else:
                # Before Risk Appetite Analysis
                st.subheader('Before Risk Appetite')

                if 'inherent_risk_rating' in filtered_data.columns:
                    risk_rating_counts = filtered_data['inherent_risk_rating'].value_counts()

                    critical_count = risk_rating_counts.get('Critical', 0)
                    severe_count = risk_rating_counts.get('High', 0)
                    moderate_count = risk_rating_counts.get('Moderate', 0)
                    sustainable_count = risk_rating_counts.get('Low', 0)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Critical Inherent Risks", critical_count)
                    with col2:
                        st.metric("High Inherent Risks", severe_count)
                    with col3:
                        st.metric("Moderate Inherent Risks", moderate_count)
                    with col4:
                        st.metric("Low Inherent Risks", sustainable_count)
                else:
                    st.warning("The column 'inherent_risk_rating' is missing from the data.")

                if 'residual_risk_rating' in filtered_data.columns:
                    residual_risk_rating_counts = filtered_data['residual_risk_rating'].value_counts()

                    residual_critical_count = residual_risk_rating_counts.get('Critical', 0)
                    residual_severe_count = residual_risk_rating_counts.get('High', 0)
                    residual_moderate_count = residual_risk_rating_counts.get('Moderate', 0)
                    residual_sustainable_count = residual_risk_rating_counts.get('Low', 0)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Critical Residual Risks", residual_critical_count)
                    with col2:
                        st.metric("High Residual Risks", residual_severe_count)
                    with col3:
                        st.metric("Moderate Residual Risks", residual_moderate_count)
                    with col4:
                        st.metric("Low Residual Risks", residual_sustainable_count)
                else:
                    st.warning("The column 'residual_risk_rating' is missing from the data.")

                # Optionally call your styling function if defined
                style_metric_cards(border_left_color="#DBF227")

                if 'risk_type' in filtered_data.columns:
                    risk_type_counts = filtered_data['risk_type'].value_counts()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(risk_type_counts.index, risk_type_counts.values, color='skyblue')
                    ax.set_title("Risk Types Count")
                    ax.set_ylabel("Count")
                    ax.set_xticklabels(risk_type_counts.index, rotation=45)

                    for bar in bars:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2, yval - 0.5, yval, ha='center', va='bottom', color='black')

                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("The column 'risk_type' is missing from the data.")
                    
                # Display Risk Dashboard
                st.subheader("Risk Count by Risk Type")
                risk_dashboard = create_risk_dashboard(risk_data)
                st.dataframe(risk_dashboard)
                
                # Prepare data for download (unstyled data)
                csv = risk_dashboard.to_csv(index=False)
                current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')

                # Provide a download button for the CSV file
                st.download_button(
                    label="Download Rish Dashboard",
                    data=csv,
                    file_name=f"risk_dashboard_{current_datetime}.csv",
                    mime="text/csv",
                )

                # After Risk Appetite Analysis
                st.subheader('After Risk Appetite')
                              
                # Check for required columns before applying further filtering
                if 'inherent_risk_rating' in filtered_data.columns and 'residual_risk_rating' in filtered_data.columns and 'risk_type' in filtered_data.columns:
                    filtered_data['risk_appetite'] = filtered_data['risk_type'].apply(get_risk_appetite)
                    
                    def residual_exceeds_appetite(row):
                        # Define a mapping of risk levels for comparison purposes
                        risk_levels = ['Low', 'Moderate', 'High', 'Critical']

                        # Check if risk appetite is empty
                        if not row['risk_appetite']:
                            return False  # or True if you want to keep risks with no defined appetite

                        # Find the maximum level in the appetite for comparison
                        max_appetite_level = max(row['risk_appetite'], key=lambda level: risk_levels.index(level))

                        # Check if residual risk rating exceeds the maximum appetite level
                        exceeds_appetite = risk_levels.index(row['residual_risk_rating']) > risk_levels.index(max_appetite_level)

                        # Check if the risk is flagged as an opportunity
                        accepted_due_to_opportunity = row['opportunity_type'] == 'Yes'

                        # If it exceeds the appetite but is accepted due to an opportunity, consider it acceptable
                        if exceeds_appetite and accepted_due_to_opportunity:
                            return False
                        return exceeds_appetite

                    risk_register = filtered_data[filtered_data.apply(residual_exceeds_appetite, axis=1)]

                    risk_rating_counts = risk_register['inherent_risk_rating'].value_counts()

                    critical_count = risk_rating_counts.get('Critical', 0)
                    severe_count = risk_rating_counts.get('High', 0)
                    moderate_count = risk_rating_counts.get('Moderate', 0)
                    sustainable_count = risk_rating_counts.get('Low', 0)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Critical Inherent Risks", critical_count)
                    with col2:
                        st.metric("High Inherent Risks", severe_count)
                    with col3:
                        st.metric("Moderate Inherent Risks", moderate_count)
                    with col4:
                        st.metric("Low Inherent Risks", sustainable_count)

                    residual_risk_rating_counts = risk_register['residual_risk_rating'].value_counts()

                    residual_critical_count = residual_risk_rating_counts.get('Critical', 0)
                    residual_severe_count = residual_risk_rating_counts.get('High', 0)
                    residual_moderate_count = residual_risk_rating_counts.get('Moderate', 0)
                    residual_sustainable_count = residual_risk_rating_counts.get('Low', 0)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Critical Residual Risks", residual_critical_count)
                    with col2:
                        st.metric("High Residual Risks", residual_severe_count)
                    with col3:
                        st.metric("Moderate Residual Risks", residual_moderate_count)
                    with col4:
                        st.metric("Low Residual Risks", residual_sustainable_count)

                    # Optionally call your styling function if defined
                    style_metric_cards(border_left_color="#DBF227")

                    if 'risk_type' in risk_register.columns:
                        risk_type_counts = risk_register['risk_type'].value_counts()

                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(risk_type_counts.index, risk_type_counts.values, color='skyblue')
                        ax.set_title("Risk Types Count")
                        ax.set_ylabel("Count")
                        ax.set_xticklabels(risk_type_counts.index, rotation=45)

                        for bar in bars:
                            yval = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2, yval - 0.5, yval, ha='center', va='bottom', color='black')

                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("The column 'risk_type' is missing from the risk register data.")
                else:
                    st.warning("The necessary columns for 'inherent_risk_rating' or 'residual_risk_rating' are missing from the data.")
                    
                
        elif tab == 'Risks Owners & Control Owners':
            st.markdown("""
            <style>
                body .stMetric span:first-child {
                    font-size: 12px !important; 
                }
                body .stMetric span:last-child {
                    font-size: 16px !important;
                }
            </style>
            """, unsafe_allow_html=True)
            
            engine = connect_to_db()

            if 'risk_data' not in st.session_state:
                st.session_state['risk_data'] = fetch_all_from_risk_data(engine)

            st.subheader('Risks Owners & Control Owners')
            
            st.subheader('Date Filters')
           
            # Load data from session state
            risk_data = st.session_state.get('risk_data', pd.DataFrame())

            # Initialize filtered_data as an empty DataFrame
            filtered_data = pd.DataFrame()

            # Check if the DataFrame is not empty and contains the 'date_last_updated' column
            if not risk_data.empty and 'date_last_updated' in risk_data.columns:
                # Ensure 'date_last_updated' is in datetime format, coerce errors to NaT
                risk_data['date_last_updated'] = pd.to_datetime(risk_data['date_last_updated'], errors='coerce')

                # Date filter section: ensure min_date and max_date are valid
                min_date = risk_data['date_last_updated'].min()
                max_date = risk_data['date_last_updated'].max()

                # Handle cases where min_date or max_date might be NaT
                if pd.isnull(min_date):
                    min_date = datetime.today().date()
                else:
                    min_date = min_date.date()

                if pd.isnull(max_date):
                    max_date = datetime.today().date()
                else:
                    max_date = max_date.date()

                # Use the dates in the Streamlit date input
                from_date = st.date_input('From', value=min_date, min_value=min_date, max_value=max_date)
                to_date = st.date_input('To', value=max_date, min_value=min_date, max_value=max_date)

                # Apply date filter to the data
                filtered_data = risk_data[(risk_data['date_last_updated'] >= pd.Timestamp(from_date)) &
                                          (risk_data['date_last_updated'] <= pd.Timestamp(to_date))]
            else:
                st.warning("The data is empty or missing the 'date_last_updated' column.")

            # If filtered_data is empty, provide a message
            if filtered_data.empty:
                st.info("No data available for the selected date range.")
            else:
                # Check if 'risk_owners' column exists before plotting
                if 'risk_owners' in filtered_data.columns:
                    risk_owners_counts = filtered_data['risk_owners'].value_counts()

                    fig = plt.figure(figsize=(10, 6))
                    bars = plt.bar(risk_owners_counts.index, risk_owners_counts.values, color='skyblue')
                    plt.title("Risk Owners Count")
                    plt.ylabel("Risk Count")
                    plt.xticks(rotation=45)

                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2, yval - 0.5, yval, ha='center', va='bottom', color='black')

                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("The column 'risk_owners' is missing from the data.")

                # Check if 'control_owners' column exists before plotting
                if 'control_owners' in filtered_data.columns:
                    risk_control_owners_counts = filtered_data['control_owners'].value_counts()

                    fig = plt.figure(figsize=(10, 6))
                    bars = plt.bar(risk_control_owners_counts.index, risk_control_owners_counts.values, color='skyblue')
                    plt.title("Risk Control Owners Count")
                    plt.ylabel("Risk Count")
                    plt.xticks(rotation=45)

                    for bar in bars:
                        yval = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2, yval - 0.5, yval, ha='center', va='bottom', color='black')

                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("The column 'control_owners' is missing from the data.")
            
          
        elif tab == 'Adjusted Risk Matrices':
            color_mapping = {
                "Critical": "red",
                "High": "orange",
                "Moderate": "yellow",
                "Low": "green",
                None: "white"
            }

            def plot_risk_matrix_with_axes_labels(matrix, risk_matrix, title, master_risk_matrix=None):
                fig = plt.figure(figsize=(10, 10))
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], ['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5], ['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                plt.xlim(0, 5)
                plt.ylim(0, 5)
                plt.xlabel('Impact')
                plt.ylabel('Probability')
                plt.title(title)

                nrows = 5
                ncols = 5
                axes = [fig.add_subplot(nrows, ncols, r * ncols + c + 1) for r in range(0, nrows) for c in range(0, ncols)]

                for r in range(0, nrows):
                    for c in range(0, ncols):
                        ax = axes[r * ncols + c]
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_xlim(0, 5)
                        ax.set_ylim(0, 5)

                        cell_value = risk_matrix[r, c]
                        if cell_value not in color_mapping:
                            st.write(f"Unexpected value '{cell_value}' in risk_matrix at ({r}, {c}). Using default color.")
                            cell_value = None

                        ax.set_facecolor(color_mapping[cell_value])

                        if matrix[r, c] > 0:
                            ax.text(2.5, 2.5, str(matrix[r, c]), ha='center', va='center', fontsize=10, weight='bold')

                legend_handles = [Line2D([0], [0], color=color_mapping[key], lw=4, label=key) for key in color_mapping if key is not None]
                plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))

                plt.tight_layout()
                st.pyplot(fig)
                
            st.subheader('Risks Filters')
            
            # Define colors for each risk rating
            colors = {
                'Low': 'background-color: green',
                'Moderate': 'background-color: yellow',
                'High': 'background-color: orange',
                'Critical': 'background-color: red'
            }

            # Function to apply styles
            def highlight_risk(rating):
                return colors.get(rating, '')
            
            engine = connect_to_db()
            
            # Load or fetch data
            risk_data = st.session_state.get('risk_data', fetch_all_from_risk_data(engine))

            # Initialize filtered_data as an empty DataFrame
            filtered_data = pd.DataFrame()

            # Check if the DataFrame is not empty and contains the 'date_last_updated' column
            if not risk_data.empty and 'date_last_updated' in risk_data.columns:
                # Ensure 'date_last_updated' is in datetime format, coerce errors to NaT
                risk_data['date_last_updated'] = pd.to_datetime(risk_data['date_last_updated'], errors='coerce')

                # Date filter section: ensure min_date and max_date are valid
                min_date = risk_data['date_last_updated'].min()
                max_date = risk_data['date_last_updated'].max()

                # Handle cases where min_date or max_date might be NaT
                if pd.isnull(min_date):
                    min_date = datetime.today().date()  # Set to today's date if NaT
                else:
                    min_date = min_date.date()  # Convert to datetime.date

                if pd.isnull(max_date):
                    max_date = datetime.today().date()  # Set to today's date if NaT
                else:
                    max_date = max_date.date()  # Convert to datetime.date

                # Use the dates in the Streamlit date input, ensuring they are valid
                from_date = st.date_input('From', value=min_date, min_value=min_date, max_value=max_date)
                to_date = st.date_input('To', value=max_date, min_value=min_date, max_value=max_date)

                # Apply date filter to the data, ensuring proper conversion to Timestamp
                filtered_data = risk_data[(risk_data['date_last_updated'] >= pd.Timestamp(from_date)) &
                                          (risk_data['date_last_updated'] <= pd.Timestamp(to_date))]
            else:
                st.warning("The data is empty or missing the 'date_last_updated' column.")

            # Add a selectbox for unit filtering
            selected_unit = st.selectbox('Select Unit', units)

            # Apply unit filter if not 'All'
            if selected_unit != 'All':
                filtered_data = filtered_data[(filtered_data['Unit'] == selected_unit) | (filtered_data['Unit'].isna())]
                
            # Add a filter for risk owners
            risk_owners = ['All'] + sorted(risk_data['risk_owners'].dropna().unique().tolist())
            selected_risk_owner = st.selectbox('Select Risk Owner', risk_owners)

            # Apply risk owners filter if not 'All'
            if selected_risk_owner != 'All':
                filtered_data = filtered_data[filtered_data['risk_owners'] == selected_risk_owner]

            st.subheader('Risk Data')

            # If filtered_data is empty, provide a message and set default date inputs
            if filtered_data.empty:
                st.info("No data available for the selected date range and unit.")

            else:
                # Before Risk Appetite Analysis
                st.subheader('Before Risk Appetite')

                probability_mapping = {
                    "Very Low": 1,
                    "Low": 2,
                    "Medium": 3,
                    "High": 4,
                    "Very High": 5
                }

                required_columns = [
                    'inherent_risk_probability', 'inherent_risk_impact',
                    'residual_risk_probability', 'residual_risk_impact'
                ]

                missing_columns = [col for col in required_columns if col not in filtered_data.columns]
                if missing_columns:
                    st.error(f"Missing columns in risk_data: {', '.join(missing_columns)}")
                else:
                    filtered_data['inherent_risk_probability_num'] = filtered_data['inherent_risk_probability'].map(probability_mapping)
                    filtered_data['inherent_risk_impact_num'] = filtered_data['inherent_risk_impact'].map(probability_mapping)
                    filtered_data['residual_risk_probability_num'] = filtered_data['residual_risk_probability'].map(probability_mapping)
                    filtered_data['residual_risk_impact_num'] = filtered_data['residual_risk_impact'].map(probability_mapping)

                    inherent_risk_matrix = np.empty((5, 5), dtype=object)
                    residual_risk_matrix = np.empty((5, 5), dtype=object)
                    inherent_risk_count_matrix = np.zeros((5, 5), dtype=int)
                    residual_risk_count_matrix = np.zeros((5, 5), dtype=int)

                    for _, row in filtered_data.iterrows():
                        prob_num = row.get('inherent_risk_probability_num')
                        impact_num = row.get('inherent_risk_impact_num')
                        inherent_risk_rating = row.get('inherent_risk_rating')
                        if prob_num and impact_num and inherent_risk_rating in colors:
                            inherent_risk_matrix[5 - prob_num, impact_num - 1] = inherent_risk_rating
                            inherent_risk_count_matrix[5 - prob_num, impact_num - 1] += 1

                        prob_num = row.get('residual_risk_probability_num')
                        impact_num = row.get('residual_risk_impact_num')
                        residual_risk_rating = row.get('residual_risk_rating')
                        if prob_num and impact_num and residual_risk_rating in colors:
                            residual_risk_matrix[5 - prob_num, impact_num - 1] = residual_risk_rating
                            residual_risk_count_matrix[5 - prob_num, impact_num - 1] += 1

                    master_risk_matrix = np.array([
                        ["Moderate", "High", "High", "Critical", "Critical"],
                        ["Low", "Moderate", "High", "Critical", "Critical"],
                        ["Low", "Moderate", "Moderate", "High", "Critical"],
                        ["Low", "Low", "Moderate", "High", "Critical"],
                        ["Low", "Low", "Low", "Moderate", "High"]
                    ])
                    
                    for i in range(5):
                        for j in range(5):
                            if not inherent_risk_matrix[i, j]:
                                inherent_risk_matrix[i, j] = master_risk_matrix[i, j]
                            if not residual_risk_matrix[i, j]:
                                residual_risk_matrix[i, j] = master_risk_matrix[i, j]

                    plot_risk_matrix_with_axes_labels(inherent_risk_count_matrix, inherent_risk_matrix, "Inherent Risk Matrix with Counts")
                    plot_risk_matrix_with_axes_labels(residual_risk_count_matrix, residual_risk_matrix, "Residual Risk Matrix with Counts")

                    st.subheader('After Risk Appetite')
                        
                    # Check for required columns before applying further filtering
                    if 'inherent_risk_rating' in filtered_data.columns and 'residual_risk_rating' in filtered_data.columns and 'risk_type' in filtered_data.columns:
                        filtered_data['risk_appetite'] = filtered_data['risk_type'].apply(get_risk_appetite)

                        def residual_exceeds_appetite(row):
                            # Define a mapping of risk levels for comparison purposes
                            risk_levels = ['Low', 'Moderate', 'High', 'Critical']

                            # Check if risk appetite is empty
                            if not row['risk_appetite']:
                                return False  # or True if you want to keep risks with no defined appetite

                            # Find the maximum level in the appetite for comparison
                            max_appetite_level = max(row['risk_appetite'], key=lambda level: risk_levels.index(level))

                            # Check if residual risk rating exceeds the maximum appetite level
                            exceeds_appetite = risk_levels.index(row['residual_risk_rating']) > risk_levels.index(max_appetite_level)

                            # Check if the risk is flagged as an opportunity
                            accepted_due_to_opportunity = row['opportunity_type'] == 'Yes'

                            # If it exceeds the appetite but is accepted due to an opportunity, consider it acceptable
                            if exceeds_appetite and accepted_due_to_opportunity:
                                return False
                            return exceeds_appetite

                        risk_register = filtered_data[filtered_data.apply(residual_exceeds_appetite, axis=1)]

                    risk_register['inherent_risk_probability_num'] = risk_register['inherent_risk_probability'].map(probability_mapping)
                    risk_register['inherent_risk_impact_num'] = risk_register['inherent_risk_impact'].map(probability_mapping)
                    risk_register['residual_risk_probability_num'] = risk_register['residual_risk_probability'].map(probability_mapping)
                    risk_register['residual_risk_impact_num'] = risk_register['residual_risk_impact'].map(probability_mapping)

                    inherent_risk_matrix = np.empty((5, 5), dtype=object)
                    residual_risk_matrix = np.empty((5, 5), dtype=object)
                    inherent_risk_count_matrix = np.zeros((5, 5), dtype=int)
                    residual_risk_count_matrix = np.zeros((5, 5), dtype=int)

                    for _, row in risk_register.iterrows():
                        prob_num = row.get('inherent_risk_probability_num')
                        impact_num = row.get('inherent_risk_impact_num')
                        inherent_risk_rating = row.get('inherent_risk_rating')
                        if prob_num and impact_num and inherent_risk_rating in colors:
                            inherent_risk_matrix[5 - prob_num, impact_num - 1] = inherent_risk_rating
                            inherent_risk_count_matrix[5 - prob_num, impact_num - 1] += 1

                        prob_num = row.get('residual_risk_probability_num')
                        impact_num = row.get('residual_risk_impact_num')
                        residual_risk_rating = row.get('residual_risk_rating')
                        if prob_num and impact_num and residual_risk_rating in colors:
                            residual_risk_matrix[5 - prob_num, impact_num - 1] = residual_risk_rating
                            residual_risk_count_matrix[5 - prob_num, impact_num - 1] += 1

                    for i in range(5):
                        for j in range(5):
                            if not inherent_risk_matrix[i, j]:
                                inherent_risk_matrix[i, j] = master_risk_matrix[i, j]
                            if not residual_risk_matrix[i, j]:
                                residual_risk_matrix[i, j] = master_risk_matrix[i, j]

                    plot_risk_matrix_with_axes_labels(inherent_risk_count_matrix, inherent_risk_matrix, "Inherent Risk Matrix with Counts")
                    plot_risk_matrix_with_axes_labels(residual_risk_count_matrix, residual_risk_matrix, "Residual Risk Matrix with Counts")
           
        elif tab == 'Performance Metrics':
            st.title('Performance Metrics')
            
            st.subheader('Risk Filters')

            # Main code starts here
            engine = connect_to_db()

            # Load data from session state
            risk_data = st.session_state.get('risk_data', fetch_all_from_risk_data(engine))

            # Initialize filtered_data as an empty DataFrame
            filtered_data = pd.DataFrame()

            # Check if the DataFrame is not empty and contains the 'date_last_updated' column
            if not risk_data.empty and 'date_last_updated' in risk_data.columns:
                # Ensure 'date_last_updated' is in datetime format, coerce errors to NaT
                risk_data['date_last_updated'] = pd.to_datetime(risk_data['date_last_updated'], errors='coerce')

                # Date filter section: ensure min_date and max_date are valid
                min_date = risk_data['date_last_updated'].min()
                max_date = risk_data['date_last_updated'].max()

                # Handle cases where min_date or max_date might be NaT
                if pd.isnull(min_date):
                    min_date = datetime.today().date()
                else:
                    min_date = min_date.date()

                if pd.isnull(max_date):
                    max_date = datetime.today().date()
                else:
                    max_date = max_date.date()

                # Use the dates in the Streamlit date input
                from_date = st.date_input('From', value=min_date, min_value=min_date, max_value=max_date)
                to_date = st.date_input('To', value=max_date, min_value=min_date, max_value=max_date)

                # Apply date filter to the data
                filtered_data = risk_data[(risk_data['date_last_updated'] >= pd.Timestamp(from_date)) &
                                          (risk_data['date_last_updated'] <= pd.Timestamp(to_date))]
            else:
                st.warning("The data is empty or missing the 'date_last_updated' column.")

            # Check if 'residual_risk_rating' column exists in the filtered data
            if 'residual_risk_rating' in filtered_data.columns:
                # Define residual risk ratings and add 'All' option
                residual_risk_ratings = filtered_data['residual_risk_rating'].dropna().unique().tolist()
                residual_risk_ratings.insert(0, 'All')

                # Add a selectbox for residual risk rating filtering
                selected_risk_rating = st.selectbox('Select Residual Risk Rating', residual_risk_ratings)

                # Apply residual risk rating filter if not 'All'
                if selected_risk_rating != 'All':
                    filtered_data = filtered_data[filtered_data['residual_risk_rating'] == selected_risk_rating]
            else:
                st.warning("The data is missing the 'residual_risk_rating' column.")

            # Check if 'risk_owners' column exists in the filtered data
            if 'risk_owners' in filtered_data.columns:
                # Define risk owners and add 'All' option
                risk_owners_list = filtered_data['risk_owners'].dropna().unique().tolist()
                risk_owners_list.insert(0, 'All')

                # Add a selectbox for risk owners filtering
                selected_risk_owner = st.selectbox('Select Risk Owner', risk_owners_list)

                # Apply risk owners filter if not 'All'
                if selected_risk_owner != 'All':
                    filtered_data = filtered_data[filtered_data['risk_owners'] == selected_risk_owner]
            else:
                st.warning("The data is missing the 'risk_owners' column.")

            # Display filtered data for debugging purposes
            # st.write("Filtered Data:", filtered_data)

            # Pie chart for Residual Risk Rating Distribution using the filtered data
            st.subheader("Residual Risk Rating Distribution")

            # Group the filtered data by 'residual_risk_rating' and count the occurrences
            rating_distribution = filtered_data['residual_risk_rating'].value_counts().reset_index()
            rating_distribution.columns = ['residual_risk_rating', 'count']

            if rating_distribution.empty:
                st.warning("No data available to display.")
            else:
                # Create labels that include both the residual risk rating and the count
                labels_risk = [f"{row['residual_risk_rating']} ({row['count']})" for _, row in rating_distribution.iterrows()]

                fig, ax = plt.subplots()
                ax.pie(rating_distribution['count'], labels=labels_risk, autopct=make_autopct(rating_distribution['count']), startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

            # Pie chart for Risk 'Status' Distribution using the filtered data
            st.subheader("Risk Status Distribution")

            # Group the filtered data by 'Status' and count the occurrences
            status_distribution = filtered_data['Status'].value_counts().reset_index()
            status_distribution.columns = ['Status', 'count']

            if status_distribution.empty:
                st.warning("No data available to display.")
            else:
                # Create labels that include both the risk status and the count
                labels_status = [f"{row['Status']} ({row['count']})" for _, row in status_distribution.iterrows()]

                fig, ax = plt.subplots()
                ax.pie(status_distribution['count'], labels=labels_status, autopct=make_autopct(status_distribution['count']), startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

            # Pie chart for 'Time Open' Distribution using the filtered data
            st.subheader("Time Open Distribution (Days)")

            # Group the filtered data by 'Time Open' and count the occurrences
            time_open_distribution = filtered_data['Time Open'].value_counts().reset_index()
            time_open_distribution.columns = ['Time Open', 'count']

            if time_open_distribution.empty:
                st.warning("No data available to display.")
            else:
                # Create labels that include both the 'Time Open' value and the count
                labels_time_open = [f"{row['Time Open']} ({row['count']})" for _, row in time_open_distribution.iterrows()]

                fig, ax = plt.subplots()
                ax.pie(time_open_distribution['count'], labels=labels_time_open, autopct=make_autopct(time_open_distribution['count']), startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

            if filtered_data.empty:
                st.warning("No data available after filtering.")
                
            # Pie chart for 'opportunity_type' Distribution using the filtered data
            st.subheader("Risk Category Distribution")
            
            # Group the filtered data by 'opportunity_type' and count the occurrences
            opportunity_type_distribution = filtered_data['opportunity_type'].value_counts().reset_index()
            opportunity_type_distribution.columns = ['opportunity_type', 'count']

            if opportunity_type_distribution.empty:
                st.warning("No data available to display.")
            else:
                # Map 'Yes' to 'Opportunity' and 'No' to 'Risk'
                opportunity_type_distribution['description'] = opportunity_type_distribution['opportunity_type'].map({
                    'Yes': 'Opportunities',
                    'No': 'Threats'
                })

                # Create labels that include both the description and the count
                labels_opportunity_type = [f"{row['description']} ({row['count']})" for _, row in opportunity_type_distribution.iterrows()]

                fig, ax = plt.subplots()
                ax.pie(opportunity_type_distribution['count'], labels=labels_opportunity_type, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

            if filtered_data.empty:
                st.warning("No data available after filtering.")

              
            # Display KPIs
            st.subheader("Risk Management KPIs")
            st.subheader('Critical Risks')
            # 1. Risk Reduction KPI
            risk_reduction_kpi_description = """
            **Risk Reduction KPI (%):** 
            This KPI measures the effectiveness of risk treatment plans in reducing critical risks. 
            It calculates the percentage of critical risks that have been downgraded to a less severe rating after treatment.
            """
            st.markdown(risk_reduction_kpi_description)
            st.metric("Risk Reduction KPI (%)", f"{calculate_kpis(risk_data)['Risk Reduction KPI (%)']:.2f}%")

            # 2. Action Completion KPI
            action_completion_kpi_description = """
            **Action Completion KPI (%):** 
            This KPI tracks the progress of risk treatment plans by measuring the percentage of planned risk treatment actions 
            that have been completed on time. It is an indicator of how well the risk treatment process is being managed.
            """
            st.markdown(action_completion_kpi_description)
            st.metric("Action Completion KPI (%)", f"{calculate_kpis(risk_data)['Action Completion KPI (%)']:.2f}%")

            # 3. Residual Risk KPI
            residual_risk_kpi_description = """
            **Residual Risk KPI (%):** 
            This KPI measures the proportion of risks that remain critical after treatment actions have been implemented. 
            A high percentage indicates that more work is needed to mitigate these risks effectively.
            """
            st.markdown(residual_risk_kpi_description)
            st.metric("Residual Risk KPI (%)", f"{calculate_kpis(risk_data)['Residual Risk KPI (%)']:.2f}%")
            
            st.subheader('High Risks')
            # 1. Risk Reduction KPI
            risk_reduction_kpi_description = """
            **Risk Reduction KPI (%):** 
            This KPI measures the effectiveness of risk treatment plans in reducing high risks. 
            It calculates the percentage of high risks that have been downgraded to a less severe rating after treatment.
            """
            st.markdown(risk_reduction_kpi_description)
            st.metric("Risk Reduction KPI (%)", f"{calculate_kpis_high(risk_data)['Risk Reduction KPI (%)']:.2f}%")

            # 2. Action Completion KPI
            action_completion_kpi_description = """
            **Action Completion KPI (%):** 
            This KPI tracks the progress of risk treatment plans by measuring the percentage of planned risk treatment actions 
            that have been completed on time. It is an indicator of how well the risk treatment process is being managed.
            """
            st.markdown(action_completion_kpi_description)
            st.metric("Action Completion KPI (%)", f"{calculate_kpis_high(risk_data)['Action Completion KPI (%)']:.2f}%")

            # 3. Residual Risk KPI
            residual_risk_kpi_description = """
            **Residual Risk KPI (%):** 
            This KPI measures the proportion of risks that remain high after treatment actions have been implemented. 
            A high percentage indicates that more work is needed to mitigate these risks effectively.
            """
            st.markdown(residual_risk_kpi_description)
            st.metric("Residual Risk KPI (%)", f"{calculate_kpis_high(risk_data)['Residual Risk KPI (%)']:.2f}%")
            
            # Load or fetch data (replace with your data loading logic)
            engine = connect_to_db()

            # Load or fetch updated data
            risk_data_updated = st.session_state.get('risk_data_updated', fetch_all_from_risk_data(engine))

            # Ensure that the data is loaded before proceeding
            if risk_data_updated is None or risk_data_updated.empty:
                st.error("No data available. Please check the data source.")
            else:
                # Function to calculate Adequacy KPI
                def calculate_adequacy_kpi(df):
                    total_risks = df.shape[0]
                    adequate_treatments = df[df['Adequacy'] == 'Acceptable'].shape[0]
                    adequacy_kpi = (adequate_treatments / total_risks) * 100 if total_risks > 0 else 0
                    return adequacy_kpi

                # Function to calculate Direction KPI
                def calculate_direction_kpi(df):
                    total_risks = df.shape[0]
                    improving_risks = df[df['Direction'] == 'Stable'].shape[0]
                    direction_kpi = (improving_risks / total_risks) * 100 if total_risks > 0 else 0
                    return direction_kpi

                # Calculate and display Adequacy KPI
                st.subheader('Adequacy and Direction Measures')

                adequacy_kpi_description = """
                **Adequacy KPI (%):** 
                This KPI measures the percentage of risks that have been addressed with treatment plans deemed 'Acceptable'.
                """
                st.markdown(adequacy_kpi_description)
                adequacy_kpi = calculate_adequacy_kpi(risk_data_updated)
                st.metric("Adequacy KPI (%)", f"{adequacy_kpi:.2f}%")

                # Calculate and display Direction KPI
                direction_kpi_description = """
                **Direction KPI (%):** 
                This KPI measures the percentage of risks that are, based on the current information residual risk is expected to be stable in the next twelve months , classified as 'Stable' in the 'Direction' field.
                """
                st.markdown(direction_kpi_description)
                direction_kpi = calculate_direction_kpi(risk_data_updated)
                st.metric("Direction KPI (%)", f"{direction_kpi:.2f}%")
                            
        elif tab == 'Reports':
            st.subheader('Reports')
            
            engine = connect_to_db()

            # Load or fetch data
            risk_data = st.session_state.get('risk_data', fetch_all_from_risk_data(engine))
            
            # Display Risk Treatment Progress Report
            st.subheader("Risk Treatment Progress Report")
            progress_report = generate_progress_reports(risk_data)
            st.dataframe(progress_report)

            
            generate_trend_analysis(risk_data)
      
        elif tab == 'Delete Risk':
            st.subheader('Delete Risk from Risk Data')
            
            engine = connect_to_db()
            
            if not st.session_state['risk_data'].empty:
                # Fetching all risks data
                risk_data_df = fetch_all_from_risk_data(engine)
                
                # Selecting a risk by its description
                risk_to_delete_description = st.selectbox('Select a risk to delete', risk_data_df['risk_description'].tolist())

                # Filtering the DataFrame to find the selected risk
                selected_risk = risk_data_df[risk_data_df['risk_description'] == risk_to_delete_description].iloc[0]
                
                if 'risk_type' in selected_risk:
                    st.write(f"**Risk Type:** {selected_risk['risk_type']}")
                else:
                    st.write("Risk Type not available.")

                if 'cause_consequences' in selected_risk:
                    st.write(f"**Cause:** {selected_risk['cause_consequences']}")
                else:
                    st.write("Cause not available.")

                if 'risk_owners' in selected_risk:
                    st.write(f"**Risk Owner(s):** {selected_risk['risk_owners']}")
                else:
                    st.write("Risk Owner(s) not available.")
                    
                if 'Status' in selected_risk:
                    st.write(f"**Status:** {selected_risk['Status']}")
                else:
                    st.write("Status not avaliable.")
                    
                if 'opportunity_type' in selected_risk:
                    st.write(f"**Opportunity Type:** {selected_risk['opportunity_type']}")
                else:
                    st.write("Opportunity Type not available.")
                    
                if 'Adequacy' in selected_risk:
                    st.write(f"**Adequacy of Risk Management Systems:** {selected_risk['Adequacy']}")
                else:
                    st.write("Adequacy of Risk Management Systems not available.")
                    
                if 'Direction' in selected_risk:
                    st.write(f"**Direction of Residual Risk Rating:** {selected_risk['Direction']}")
                else:
                    st.write("Direction of Residual Risk Rating is not available.")
                                 
                if st.button('Delete Risk'):
                    initial_count = len(st.session_state['risk_data'])
                    delete_from_risk_data_by_risk_description(risk_to_delete_description)
                    st.session_state['risk_data'] = fetch_all_from_risk_data(engine)
                    if len(st.session_state['risk_data']) < initial_count:
                        st.write("Risk deleted.")
            else:
                st.write("No risks to delete.")
         
        elif tab == 'Delete/Update User':
            st.subheader('Delete and Update Users')
            
            # Assuming you have already established a database connection
            engine = connect_to_db()

            # Function to retrieve user information (username and role) from the credentials table
            def get_user_info(username):
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT user, role FROM credentials WHERE user = :username"), {'username': username}).fetchone()
                    if result:
                        # Accessing by positional indices
                        return result[0], result[1]
                    return None, None

            # Function to retrieve all usernames from the credentials table
            def get_all_usernames():
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT user FROM credentials")).fetchall()
                    return [row[0] for row in result]

            # Function to retrieve distinct roles from the credentials table
            def get_all_roles():
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT DISTINCT role FROM credentials")).fetchall()
                    return [row[0] for row in result]

            # Assuming the logged-in user is in session state
            logged_in_user = st.session_state.user

            # Retrieve the current user's name and role from the credentials table
            current_user, current_role = get_user_info(logged_in_user)

            # Fetch the list of all usernames and roles from the database
            all_usernames = get_all_usernames()
            all_roles = get_all_roles()

            # Tab logic
            tab = st.selectbox("Select Option", ["Delete User", "Update User"])

            # Check if the current user is superadmin
            if current_role == "superadmin":

                # DELETE USER LOGIC
                if tab == "Delete User":
                    st.subheader("Delete User")
                    delete_user = st.selectbox("Select username to delete", all_usernames)  # Dynamically populate the usernames
                    if st.button("Delete"):
                        try:
                            with engine.connect() as conn:
                                conn.execute(text(f"DELETE FROM credentials WHERE user = :user"), {'user': delete_user})
                                st.success(f"User {delete_user} deleted successfully.")
                        except Exception as e:
                            st.error(f"Error deleting user: {e}")

                # UPDATE USER LOGIC
                elif tab == "Update User":
                    st.subheader("Update User")

                    # Select user to update
                    update_user = st.selectbox("Select username to update", all_usernames)  # Dynamically populate the usernames

                    # Fetch the current role of the selected user
                    selected_user, selected_role = get_user_info(update_user)

                    # Display the user's current role in the dropdown and allow it to be changed
                    new_role = st.selectbox("Select new role", all_roles, index=all_roles.index(selected_role))  # Set default to current role

                    # Select new expiry date
                    new_expiry_date = st.date_input("Select new expiry date")

                    if st.button("Update"):
                        try:
                            with engine.connect() as conn:
                                conn.execute(text(f"UPDATE credentials SET role = :role, expiry_date = :expiry WHERE user = :user"), 
                                             {'role': new_role, 'expiry': new_expiry_date, 'user': update_user})
                                st.success(f"User {update_user} updated successfully.")
                        except Exception as e:
                            st.error(f"Error updating user: {e}")
            else:
                st.warning("You do not have permission to delete or update users.")
                             
        elif tab == 'Update Risk':
            st.subheader('Update Risk in Risk Data')
            
            engine = connect_to_db()
            
            # Fetch the current 'risk_type' from the selected row
            if not st.session_state['risk_data'].empty:
                # Fetch the risk descriptions for selection
                risk_to_update = st.selectbox('Select a risk to update', fetch_all_from_risk_data(engine)['risk_description'].tolist())

                # Filter the DataFrame for the selected risk description
                filtered_risk_data = st.session_state['risk_data'][st.session_state['risk_data']['risk_description'] == risk_to_update]

                if not filtered_risk_data.empty:
                    # Select the row corresponding to the selected risk description
                    selected_risk_row = filtered_risk_data.iloc[0]
                    
                    # Define the list of risk types once
                    risk_types = sorted([
                        'Strategic Risk', 'Operational Risk', 'Compliance Risk', 'Reputational Risk', 'Financial Risk',
                        'Regulatory Risk', 'Environmental Risk', 'Human Resource Risk',
                        'Supply Chain Risk', 'Ethical Risk', 'Technological Risk', 'Public Health Risk'
                    ])

                    # Allow user to change the 'risk_type' with the current value pre-selected
                    st.session_state['risk_type'] = st.selectbox('Risk Type', risk_types, 
                        index=risk_types.index(selected_risk_row['risk_type']))
                   
                    # Display fields for updating the risk
                    updated_risk_description = st.text_input('risk_description', value=selected_risk_row['risk_description'])
                    updated_cause_consequences = st.text_input('cause_consequences', value=selected_risk_row['cause_consequences'])
                    updated_risk_owners = st.text_input('risk_owners', value=selected_risk_row['risk_owners'])
                    updated_inherent_risk_probability = st.selectbox('inherent_risk_probability', list(risk_levels.keys()), index=list(risk_levels.keys()).index(selected_risk_row['inherent_risk_probability']))
                    updated_inherent_risk_impact = st.selectbox('inherent_risk_impact', list(risk_levels.keys()), index=list(risk_levels.keys()).index(selected_risk_row['inherent_risk_impact']))
                    updated_controls = st.text_input('controls', value=selected_risk_row['controls'])

                    # New field for updating Adequacy
                    updated_adequacy = st.selectbox('Adequacy', ['Weak', 'Acceptable', 'Strong'], index=['Weak', 'Acceptable', 'Strong'].index(selected_risk_row['Adequacy']))

                    updated_control_owners = st.text_input('control_owners', value=selected_risk_row['control_owners'])
                    updated_residual_risk_probability = st.selectbox('residual_risk_probability', list(risk_levels.keys()), index=list(risk_levels.keys()).index(selected_risk_row['residual_risk_probability']))
                    updated_residual_risk_impact = st.selectbox('residual_risk_impact', list(risk_levels.keys()), index=list(risk_levels.keys()).index(selected_risk_row['residual_risk_impact']))

                    # New field for updating Direction
                    updated_direction = st.selectbox('Direction', ['Increasing', 'Decreasing', 'Stable'], index=['Increasing', 'Decreasing', 'Stable'].index(selected_risk_row['Direction']))

                    updated_by = st.text_input('updated_by', value=selected_risk_row['updated_by'])
                    updated_date_last_updated = st.date_input('date_last_updated', value=selected_risk_row['date_last_updated'])
                    
                    # Define the list of units once
                    units = sorted([
                        'Licensing and Enforcement', 'Evaluations and Registration', 'Pharmacovigilance and Clinical Trials',
                        'Chemistry Laboratory', 'Microbiology Laboratory', 'Medical Devices Laboratory', 'Quality Unit',
                        'Legal Unit', 'Human Resources', 'Information and Communication Technology', 'Finance and Administration'
                    ])

                    # New field for updating unit
                    updated_unit = st.selectbox('Unit', units, index=units.index(selected_risk_row['Unit']))

                    # New field for updating Status
                    updated_status = st.selectbox('Status', ['Open', 'Closed'], index=['Open', 'Closed'].index(selected_risk_row['Status']))

                    # New field for updating Opportunity Type
                    updated_opportunity_type = st.selectbox('Is there an Opportunity associated with this risk?', ['No', 'Yes'], index=['No', 'Yes'].index(selected_risk_row.get('opportunity_type', 'No')))

                    if st.button('Update Risk'):
                        updated_risk = {
                        'risk_type': st.session_state['risk_type'],
                        'updated_by': updated_by,
                        'date_last_updated': updated_date_last_updated.strftime('%Y-%m-%d'),
                        'risk_description': updated_risk_description,
                        'cause_consequences': updated_cause_consequences,
                        'risk_owners': updated_risk_owners,
                        'inherent_risk_probability': updated_inherent_risk_probability,
                        'inherent_risk_impact': updated_inherent_risk_impact,
                        'inherent_risk_rating': calculate_risk_rating(updated_inherent_risk_probability, updated_inherent_risk_impact),
                        'controls': updated_controls,
                        'Adequacy': updated_adequacy,  # Capitalize key to match database field
                        'control_owners': updated_control_owners,
                        'residual_risk_probability': updated_residual_risk_probability,
                        'residual_risk_impact': updated_residual_risk_impact,
                        'residual_risk_rating': calculate_risk_rating(updated_residual_risk_probability, updated_residual_risk_impact),
                        'Direction': updated_direction,  # Capitalize key to match database field
                        'Unit': updated_unit,
                        'Status': updated_status,
                        'opportunity_type': updated_opportunity_type
                    }
                        old_data = st.session_state['risk_data'].copy()
                        update_risk_data_by_risk_description(risk_to_update, updated_risk)
                        st.session_state['risk_data'] = fetch_all_from_risk_data(engine)
                        if not old_data.equals(st.session_state['risk_data']):
                            st.write("Risk updated.")
                else:
                    st.write("No matching risk found to update.")
            else:
                st.write("No risks to update.")
                
if __name__ == '__main__':
    main()
        

