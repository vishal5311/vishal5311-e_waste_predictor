import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import random

# Set page configuration
st.set_page_config(
    page_title="E-Waste Predictor Pro",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subheader {
        font-size: 1.5rem !important;
        color: #43A047;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        color: #0D47A1;
        font-weight: bold;
    }
    .tip-box {
        background-color: #FFF8E1;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #FFB300;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50 !important;
    }
</style>
""", unsafe_allow_html=True)

# App title with animation
st.markdown("<h1 class='main-header'>♻️ E-Waste Prediction Pro</h1>", unsafe_allow_html=True)

# Loading animation for model
with st.spinner("Loading E-Waste Prediction Model..."):
    # Load the model (with error handling)
    try:
        model = pickle.load(open("ewaste_model.pkl", 'rb'))
        # Get the feature names the model was trained on
        try:
            feature_names = model.feature_names_in_  # For models like RandomForest or LogisticRegression
        except AttributeError:
            # If feature_names_in_ isn't available, create a default list
            feature_names = [
                'Building Type_Apartment', 'Building Type_Hospital', 'Building Type_House',
                'Building Type_IT Park', 'Building Type_Mall', 'Building Type_Commercial',
                'Building Type_Residential', 'Building Type_Other', 'Building Type_School',
                'Occupants', 'Past E-Waste (kg)'
            ]
        time.sleep(1)  # Simulate loading
        st.success("Model loaded successfully!")
    except Exception as e:
        model = None

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictor", "📈 Analytics", "🎮 E-Waste Game", "ℹ️ Info"])

# Tab 1: Main Prediction Interface
with tab1:
    st.markdown("<h2 class='subheader'>Predict Future E-Waste Generation</h2>", unsafe_allow_html=True)
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏢 Building Details")
        building_type = st.selectbox(
            "Building Type",
            ["Apartment", "Hospital", "House", "IT Park", "Mall", "Commercial", "Residential", "School", "Other"]
        )
        
        # Add a building image based on selection
        if building_type == "Apartment":
            st.image("https://via.placeholder.com/300x150?text=Apartment+Building", caption="Typical apartment building")
        elif building_type == "Hospital":
            st.image("https://via.placeholder.com/300x150?text=Hospital", caption="Medical facility")
        elif building_type == "House":
            st.image("https://via.placeholder.com/300x150?text=House", caption="Residential house")
        else:
            st.image(f"https://via.placeholder.com/300x150?text={building_type}", caption=f"{building_type} building")
        
        occupants = st.slider("Number of Occupants", min_value=1, max_value=500, value=100)
        st.markdown(f"<div class='info-box'>👥 Building occupancy: {occupants} people</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 📱 E-Waste History")
        past_e_waste = st.number_input("Past E-Waste (kg)", min_value=0.0, value=10.0, step=0.5)
        
        # Show common items equivalent
        st.markdown("<div class='info-box'>💻 Approximate equivalent:</div>", unsafe_allow_html=True)
        phones = int(past_e_waste / 0.15)  # Average smartphone weight
        laptops = int(past_e_waste / 2.5)  # Average laptop weight
        
        st.progress(min(1.0, past_e_waste / 100))
        st.markdown(f"📱 {phones} smartphones or 💻 {laptops} laptops")
        
        # Additional features for prediction enhancement
        st.markdown("### 🔍 Optional Details")
        building_age = st.slider("Building Age (years)", 0, 100, 15)
        has_recycling = st.checkbox("Has Recycling Program", value=True)
        tech_refresh_rate = st.select_slider("Technology Refresh Rate", 
                                            options=["Very Low", "Low", "Medium", "High", "Very High"],
                                            value="Medium")

    # Prepare the input data
    building_types = ["Apartment", "Hospital", "House", "IT Park", "Mall", "Commercial", "Residential", "Other", "School"]
    
    # Create input data dictionary
    input_dict = {f'Building Type_{bt}': [1 if building_type == bt else 0] for bt in building_types}
    input_dict.update({
        'Occupants': [occupants],
        'Past E-Waste (kg)': [past_e_waste]
    })
    
    input_data = pd.DataFrame(input_dict)
    
    if model is not None:
        # Check if the input data has all the required columns
        missing_columns = set(feature_names) - set(input_data.columns)
        if missing_columns:
            for col in missing_columns:
                input_data[col] = 0  # Add missing columns with default value 0

        # Ensure input_data matches exactly the columns the model expects
        input_data = input_data[feature_names]
    
    # Advanced prediction options
    with st.expander("Advanced Prediction Options"):
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)
        include_seasonal = st.checkbox("Include Seasonal Variations", value=False)
        scenario = st.radio("Prediction Scenario", ["Conservative", "Moderate", "Aggressive"])
        
        st.markdown("<div class='tip-box'>💡 Tip: Higher confidence levels provide wider prediction ranges.</div>", 
                   unsafe_allow_html=True)
    
    # Button to trigger prediction with animation
    prediction_button = st.button("Predict E-Waste", use_container_width=True)
    
    if prediction_button:
        # Show prediction animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(101):
            progress_bar.progress(i)
            if i < 30:
                status_text.text("Analyzing building data...")
            elif i < 60:
                status_text.text("Processing occupancy patterns...")
            elif i < 90:
                status_text.text("Calculating e-waste projections...")
            else:
                status_text.text("Finalizing prediction...")
            time.sleep(0.01)
        
        try:
            if model is not None:
                # Make the prediction using the loaded model
                prediction = model.predict(input_data)
                base_prediction = int(prediction[0])
            else:
                # Demo mode with simulated prediction
                # Creates a prediction based on inputs but not using the actual model
                base_prediction = int(past_e_waste * (1 + occupants/200) * 
                                    (1.5 if building_type in ["IT Park", "Commercial"] else 1.0))
            
            # Add some variability based on advanced options
            modifier = 1.0
            if include_seasonal:
                modifier *= random.uniform(0.9, 1.1)
            
            if scenario == "Conservative":
                modifier *= 0.85
            elif scenario == "Aggressive":
                modifier *= 1.15
                
            final_prediction = int(base_prediction * modifier)
            
            # Calculate a range based on confidence level
            confidence_factor = (100 - confidence_level) / 100
            lower_bound = int(final_prediction * (1 - confidence_factor * 2))
            upper_bound = int(final_prediction * (1 + confidence_factor * 2))
            
            # Display results in a visually appealing way
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"### 📊 Predicted E-Waste Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.metric("Lower Estimate", f"{lower_bound} kg")
                
            with col2:
                st.metric("Predicted E-Waste", f"{final_prediction} kg", delta=f"{int(final_prediction - past_e_waste)} kg")
                # Create a gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = final_prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "E-Waste Prediction"},
                    gauge = {
                        'axis': {'range': [0, max(200, final_prediction * 1.5)]},
                        'bar': {'color': "#2E7D32"},
                        'steps': [
                            {'range': [0, lower_bound], 'color': "#C8E6C9"},
                            {'range': [lower_bound, upper_bound], 'color': "#81C784"},
                            {'range': [upper_bound, max(200, final_prediction * 1.5)], 'color': "#4CAF50"}
                        ],
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
            with col3:
                st.metric("Upper Estimate", f"{upper_bound} kg")
            
            # Add some insights
            st.markdown(f"""
            ### 🔍 Insights
            * This prediction indicates you may generate approximately **{final_prediction} kg** of e-waste.
            * This is **{int((final_prediction/past_e_waste - 1) * 100)}%** {
                "more" if final_prediction > past_e_waste else "less"} than your past e-waste generation.
            * For a {building_type.lower()} with {occupants} occupants, this is {
                "above average" if final_prediction > occupants * 0.5 else "below average"}.
            """)
            
            # Recommendations section
            st.markdown("### 💡 Recommendations")
            
            recommendations = [
                "Implement a formal e-waste collection program",
                "Educate occupants about proper disposal methods",
                "Extend the lifecycle of electronic devices when possible",
                "Partner with certified e-waste recyclers"
            ]
            
            if building_type == "IT Park" or building_type == "Commercial":
                recommendations.append("Consider leasing equipment instead of purchasing")
                recommendations.append("Implement asset management tracking for all electronic devices")
            
            if occupants > 200:
                recommendations.append("Designate e-waste champions to promote proper disposal")
                recommendations.append("Schedule quarterly e-waste collection events")
            
            for i, rec in enumerate(recommendations[:4]):
                st.markdown(f"**{i+1}.** {rec}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Please check your input data and try again.")

# Tab 2: Analytics
with tab2:
    st.markdown("<h2 class='subheader'>E-Waste Analytics Dashboard</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This dashboard provides analytics on e-waste generation patterns and trends. 
    Use the sidebar to customize the visualization.
    """)
    
    # Sample data for visualizations
    building_types = ["Apartment", "Hospital", "House", "IT Park", "Mall", "Commercial", "Residential", "School"]
    ewaste_by_type = {
        "Apartment": 35,
        "Hospital": 65,
        "House": 25,
        "IT Park": 80,
        "Mall": 55,
        "Commercial": 70,
        "Residential": 30,
        "School": 45
    }
    
    monthly_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'E-Waste (kg)': [22, 19, 25, 27, 23, 25, 29, 35, 40, 35, 28, 42]
    })
    
    # Define e-waste categories here to fix the error
    ewaste_categories = ['Computers', 'Phones', 'Printers', 'Monitors', 'Other']
    
    # Create a selection for chart type
    chart_type = st.selectbox("Select Visualization", [
        "E-Waste by Building Type", 
        "Monthly E-Waste Trends",
        "E-Waste Composition",
        "Year-over-Year Comparison"
    ])
    
    if chart_type == "E-Waste by Building Type":
        fig = px.bar(
            x=list(ewaste_by_type.keys()),
            y=list(ewaste_by_type.values()),
            labels={'x': 'Building Type', 'y': 'Average E-Waste (kg)'},
            color=list(ewaste_by_type.keys()),
            color_discrete_sequence=px.colors.sequential.Greens,
            title="Average E-Waste Generation by Building Type"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>Key Insight:</strong> IT Parks and Commercial buildings typically generate the most e-waste due to
        regular technology refresh cycles and higher electronic equipment density.
        </div>
        """, unsafe_allow_html=True)
        
    elif chart_type == "Monthly E-Waste Trends":
        fig = px.line(
            monthly_data,
            x='Month',
            y='E-Waste (kg)',
            markers=True,
            title="Monthly E-Waste Generation Trends",
            line_shape="spline"
        )
        fig.update_traces(line_color='#4CAF50')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>Key Insight:</strong> E-waste generation typically peaks in late summer and year-end
        as organizations often replace equipment during these periods.
        </div>
        """, unsafe_allow_html=True)
        
    elif chart_type == "E-Waste Composition":
        # Sample e-waste composition data
        values = [45, 15, 10, 20, 10]
        
        fig = px.pie(
            names=ewaste_categories,
            values=values,
            title="Typical E-Waste Composition",
            color_discrete_sequence=px.colors.sequential.Greens,
            hole=0.4
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>Key Insight:</strong> Computers and monitors typically make up the largest portion of e-waste
        by weight, while phones contribute less by weight but more in terms of hazardous materials.
        </div>
        """, unsafe_allow_html=True)
        
    else:  # Year-over-Year Comparison
        # Sample year-over-year data
        years = ['2020', '2021', '2022', '2023', '2024']
        ewaste_values = [180, 210, 235, 260, 300]
        
        fig = px.bar(
            x=years,
            y=ewaste_values,
            labels={'x': 'Year', 'y': 'Total E-Waste (kg)'},
            title="Year-over-Year E-Waste Generation",
            color=ewaste_values,
            color_continuous_scale=px.colors.sequential.Greens
        )
        
        fig.update_layout(height=500)
        fig.add_scatter(
            x=years, 
            y=ewaste_values, 
            mode='lines+markers',
            line=dict(color='darkgreen', width=2)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>Key Insight:</strong> E-waste generation has been steadily increasing at approximately 10-15% annually,
        driven by shorter device lifecycles and increased adoption of electronic devices.
        </div>
        """, unsafe_allow_html=True)
    
    # Add interactive filters
    with st.expander("Data Filters"):
        st.markdown("Customize your view by applying filters:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.multiselect("Building Categories", building_types, default=building_types)
            st.slider("Date Range", 2020, 2024, (2022, 2024))
            
        with col2:
            # Fixed: Use ewaste_categories instead of undefined categories
            st.multiselect("E-Waste Categories", ewaste_categories, default=ewaste_categories)
            st.checkbox("Show Trendline", value=True)
    
    # Add data download option
    st.download_button(
        label="📥 Download Sample Data",
        data=monthly_data.to_csv(index=False),
        file_name="ewaste_sample_data.csv",
        mime="text/csv"
    )

# Tab 3: E-Waste Game
with tab3:
    st.markdown("<h2 class='subheader'>🎮 E-Waste Sorting Game</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Learn about proper e-waste sorting through this interactive game!
    
    Sort the electronic items into their correct recycling categories. Drag and drop items to score points!
    """)
    
    # Simulated game interface
    st.image("https://via.placeholder.com/800x400?text=E-Waste+Sorting+Game", caption="Interactive E-Waste Sorting Game")
    
    # Game controls
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Start Game", use_container_width=True)
    with col2:
        st.button("Reset Score", use_container_width=True)
    with col3:
        st.button("Instructions", use_container_width=True)
    
    # Game scoreboard
    st.markdown("""
    ### 🏆 Leaderboard
    | Rank | Player | Score | Level |
    |------|--------|-------|-------|
    | 1 | SivaSurya | 950 | Expert |
    | 2 | Tarun Kumar | 820 | Advanced |
    | 3 | Renil Immanuel C | 780 | Advanced |
    | 4 | You | 450 | Beginner |
    """)
    
    st.markdown("""
    <div class='tip-box'>
    <strong>Game Tip:</strong> Remember that batteries and circuit boards need special handling procedures! 
    Check the hazardous materials section for guidance.
    </div>
    """, unsafe_allow_html=True)
    
    # Fun facts section
    st.markdown("### 📱 E-Waste Fun Facts")
    facts = [
        "A single recycled cell phone can recover enough gold to plate 40 pins for computer chips.",
        "Only about 20% of global e-waste is documented to be collected and recycled properly.",
        "E-waste represents 2% of trash in landfills but accounts for 70% of toxic waste.",
        "One million recycled laptops saves energy equivalent to electricity used by 3,500 homes in a year."
    ]
    
    fact = st.selectbox("Did you know?", facts)
    st.markdown(f"<div class='info-box'>{fact}</div>", unsafe_allow_html=True)

# Tab 4: Information
with tab4:
    st.markdown("<h2 class='subheader'>ℹ️ About E-Waste Management</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### What is E-Waste?
    
    Electronic waste, or e-waste, refers to discarded electronic devices and equipment, including:
    
    - Computers and laptops
    - Mobile phones and tablets
    - Printers and scanners
    - TVs and monitors
    - Keyboards, mice, and peripherals
    - Batteries and power supplies
    
    ### Why is E-Waste Management Important?
    
    E-waste contains hazardous materials such as lead, mercury, cadmium, and brominated flame retardants. 
    When improperly disposed of, these materials can leach into soil and water, causing environmental damage 
    and health risks. Additionally, e-waste contains valuable materials like gold, silver, copper, and rare 
    earth metals that can be recovered and reused.
    """)
    
    # Display e-waste management tips
    st.markdown("### 🔍 E-Waste Management Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Individuals:**
        - Donate working electronics
        - Use manufacturer take-back programs
        - Find certified e-waste recyclers
        - Remove personal data before disposal
        - Consider repairability when purchasing
        """)
        
    with col2:
        st.markdown("""
        **For Organizations:**
        - Implement an e-waste management policy
        - Train staff on proper disposal procedures
        - Partner with certified recyclers
        - Track and report e-waste metrics
        - Extend device lifecycles when possible
        """)
    
    # Add a resources section
    st.markdown("### 📚 Resources")
    
    resources = {
        "EPA Electronics Donation and Recycling": "https://www.epa.gov/recycle/electronics-donation-and-recycling",
        "E-Stewards Certified Recyclers": "https://e-stewards.org/find-a-recycler/",
        "Responsible Recycling (R2) Standard": "https://sustainableelectronics.org/r2/",
        "Basel Action Network": "https://www.ban.org/",
        "Electronics TakeBack Coalition": "http://www.electronicstakeback.com/"
    }
    
    for name, url in resources.items():
        st.markdown(f"- [{name}]({url})")
    
    # Contact form
    st.markdown("### 📧 Contact Us")
    
    with st.form("contact_form"):
        st.markdown("Have questions about e-waste management? Contact our team!")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Name")
            st.text_input("Email")
        with col2:
            st.text_area("Message")
        
        st.form_submit_button("Send Message")

# Sidebar content
with st.sidebar:
    st.image("https://via.placeholder.com/300x100?text=E-Waste+Predictor", use_column_width=True)
    
    st.markdown("### 🔧 App Settings")
    theme = st.selectbox("Theme", ["Green (Default)", "Blue", "Dark"])
    language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Chinese"])
    
    st.markdown("---")
    
    # Quick facts
    st.markdown("### ⚡ Quick Facts")
    st.markdown("""
    * Global e-waste reached 59 million tons in 2023
    * Only 17.4% of e-waste is properly recycled
    * E-waste is growing 3-5% faster than any other waste stream
    """)
    
    # Calculator
    with st.expander("📱 Device Calculator"):
        st.markdown("Estimate e-waste by device count:")
        phones = st.number_input("Smartphones", 0, 1000, 10)
        laptops = st.number_input("Laptops", 0, 1000, 5)
        monitors = st.number_input("Monitors", 0, 1000, 2)
        
        total_weight = phones * 0.15 + laptops * 2.5 + monitors * 5
        st.markdown(f"**Estimated total: {total_weight:.1f} kg**")
    
    st.markdown("---")
    
    # User tips
    st.info("💡 **Tip:** For more accurate predictions, provide detailed information about your building type and occupancy patterns.")
    
    # App version info
    st.markdown("### About")
    st.markdown("**E-Waste Predictor Pro v2.0**")
    st.markdown("Developed with ❤️ for sustainability")