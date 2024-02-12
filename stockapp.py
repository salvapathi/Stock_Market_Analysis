import os #
import streamlit as st
import subprocess
dependies_path=r"C:\Users\DELL\Downloads\scripts"
os.chdir(dependies_path)
exec(open(r"C:\Users\DELL\Downloads\scripts\source_code.py").read()) 

def main():
    st.title("Stock Analysis App")
    data = None
    column="Close"
    window_size=5
    span=3
    window_size_short_term_sma=50
    window_size_long_term_sma=100
    window_size_short_term_ema=50
    window_size_long_term_ema=100
    # Data fetching
    data_fetching = st.radio("Do you have Excel data?", ("Yes", "No"))
    if data_fetching == "No":
        start_date_str = st.text_input("Enter the start date (format: YYYY-MM-DD): ")
        end_date_str = st.text_input("Enter the end date (format: YYYY-MM-DD): ")
        stock_symbol = st.text_input("Enter the Stock symbol of particular stock: ").upper()
        if st.button("Fetch Data"):
            data = Stock_data(stock_symbol, start_date_str, end_date_str)
            
        
    else:
        file_link = st.text_input("Enter the file path:")
        column = st.selectbox("Select column for analysis:", ["Close", "Open", "High", "Low"])
        if st.button("Extract Data"):
            data = data_extraction(file_link, column)
            
    # Data wrangling
    data=data_wrandling(data)
    stock_description(data,column)
    get_pe_ratio(stock_symbol)
    data=calculating_avgs(data, column, window_size, span)

    st.subheader("Simple Moving Average (SMA) Graph Over Price")
    fig_sma= graph_simple_removing_avg(data)
    st.plotly_chart(fig_sma, use_container_width=False)
    
  
    st.subheader("Exponential Moving Average Graph")
    fig_ema = graph_exponential_moving_avg(data)
    st.subheader("Short-Term and Long-Term Simple Moving Averages Price Comparison ")
    fig_smals = long_short_SME(data,"Close",window_size_short_term_sma,window_size_long_term_sma,"Short_term_SSE","Long_Term_SSE")
    st.subheader("Short-Term and Long-Term Exponential Moving Average (EMA) Price Comparison ")
    fig_exls=long_short_EMA(data, column, window_size_short_term_ema,window_size_long_term_ema,"Short_term_EMA","Long_Term_EMA")
    st.subheader("Relative_Strength_index for Bear & BUll Signals")
    gig_rsi=Relative_Strength_index(data)
    st.subheader("Volume_Analysis")
    volume_analysis(data)
    st.subheader("Supports and Resistance")
    resistance_supports(data)

# if __name__ == "__main__":
# #     main()
# # Install Ngrok authtoken
#     subprocess.run(["C:/Users/DELL/AppData/Roaming/npm/ngrok.cmd", "authtoken", "2bOwKAOSTAgqQS1iBhjIuqwyGOa_4KtDkteXofC4VsB1YbHvg"])

#     # Use Ngrok to expose the Streamlit app to the internet
#     subprocess.Popen(["C:/Users/DELL/AppData/Roaming/npm/ngrok.cmd", "http", "8501"])

#     # Run the Streamlit app
#     main()

import subprocess

try:
    print("Terminating active ngrok session...")
    subprocess.run(["C:/Users/DELL/AppData/Roaming/npm/ngrok.cmd", "kill"])

    print("Installing Ngrok authtoken...")
    subprocess.run(["C:/Users/DELL/AppData/Roaming/npm/ngrok.cmd", "authtoken", "2c7p2kLhZ8kcRfIsbSg1lzVWOqV_84ytnNJmQVngJaowK2CvZ"])

    print("Starting ngrok to expose the Streamlit app...")
    ngrok_process = subprocess.Popen(["C:/Users/DELL/AppData/Roaming/npm/ngrok.cmd", "http", "8504"], stdout=subprocess.PIPE)

    # Read the console output line by line
    for line in ngrok_process.stdout:
        line = line.decode("utf-8").strip()
        if line.startswith("Forwarding"):
            # Extract and print the Ngrok URL
            ngrok_url = line.split()[1]
            print("Ngrok URL:", ngrok_url)
            break  # Stop reading the output once the Ngrok URL is found

except Exception as e:
    print("An error occurred:", e)


# Run the Streamlit app
main()


