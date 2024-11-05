import streamlit as st; import pandas as pd
import Pages.data_manipulation as data_manipulation
import Pages.data_simulating as data_simulating
import Pages.multivariate_visualization as multivariate_visualization
import Pages.hypothesis_testing as hypothesis_testing
import Pages.manova as manova
st.set_page_config(page_title="Homepage",page_icon="🏚️",)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", [
    "Homepage", 
    "Data Manipulation", 
    "Data Simulating", 
    "Multivariate Visualization", 
    "Hypothesis Testing", 
    "MANOVA"
])

def show():
    # ส่วนแสดงผลของหน้า
    ## ฟังก์ชันแสดงข้อความบนหน้าเว็บ
    def Homepage_show():
        head = """ <h1 style='text-align: center'> Welcome to Multivariate Analysis web application! 👋 </h1> """
        st.markdown(head, unsafe_allow_html=True)
        st.write("# ") 
    
        ### คำอธิบายเว็บแอป
        st.markdown(""" **Description:** This web application has been developed as part of the Multivariate Analysis course.
            It aims to provide a comprehensive platform for visualizing, analyzing and hypothesis testing for multivariate data.
            **Before next step, please upload your file. Drop it below 👇** """ )
    Homepage_show()
    
    # ส่วนตัวอย่างของข้อมูล
    st.markdown('## Sample of data for upload.')
    st.write("This data is from Swiss dataset, please upload your data in a format that's similar to this sample data. ")
    swiss = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/swiss.csv')
    st.write(swiss)
    st.write( """ Note: Or you want use this dataset, just click here 👇. """ )
    if st.button("Save Swiss dataset for next pages",key='Save_button1'):
            swiss = swiss.select_dtypes(include=['number'])
            st.session_state.swiss = swiss 
    st.markdown('---')  
    
    # ส่วนรับไฟล์
    ## บล็อคอัพโหลดไฟล์
    st.markdown('## Upload file type .xlsx or .csv')
    uploaded_file = st.file_uploader("Choose your .xlsx or .csv file", type=['xlsx','csv']) # สร้างบล็อคอัพโหลดไฟล์ .xlsx และ .csv
    
    ## ถ้าเกิดว่าอัปโหลดไฟล์แล้วจะเข้าเงื่อนไขนี้
    if uploaded_file:
        ### เก็บไฟล์ไว้ใน session_state
        st.session_state.uploaded_file = uploaded_file 
        ### อ่านข้อมูลไฟล์ว่าไฟล์ เป็นไฟล์ประเภทไหน
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        ### .name คือ เอาชื่อของไฟล์ .endswith คือ เอาคำลงท้าย นั่นคือ ตรวจสอบไฟล์แล้วเอาแค่ชื่อส่วนท้ายของมัน
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        ### ถ้าไม่อัปโหลดไฟล์ .xlsx หรือ .csv file จะขึ้นเตือน
        else:
            st.write("Please Upload file type .xlsx or .csv!")
        st.session_state.df = df      
    
        # หาคอลัมน์ที่เหมาะสมสำหรับจัดกลุ่ม
        grouping_column = None
        for col in df.columns:
            unique_vals = df[col].nunique()
            if 1 < unique_vals < len(df) // 2:  # เงื่อนไขทั่วไปในการเลือกคอลัมน์จัดกลุ่ม
                grouping_column = col
                break  
    
         # ส่วนตรวจสอบและจัดกลุ่มข้อมูล
        if grouping_column:
            st.write(f"Grouping column `{grouping_column}` automatically detected. Reshaping data...")
    
            # แยกข้อมูลตามค่าในคอลัมน์จัดกลุ่มและปรับรูปแบบข้อมูล
            reshaped_df = pd.concat([df[df[grouping_column] == i].reset_index(drop=True).drop(columns=[grouping_column]) 
                 for i in sorted(df[grouping_column].unique())],axis=1)
    
            # เปลี่ยนชื่อคอลัมน์ให้เป็นรูปแบบไดนามิก
            reshaped_df.columns = [f"{col}_{i}" for i in sorted(df[grouping_column].unique()) for col in df.columns if col != grouping_column]
            st.session_state.df = reshaped_df 
    
        
    # ส่วนตรวจสอบไฟล์  
    ## เช็คเงื่อนไขว่า df อยู่ใน session_state ใช่หรือไม่ ถ้าอยู่ดึงค่า ถ้าไม่คืนค่า None
      ### เพิ่มเติมเพราะว่า ถ้าเราไปใช้หน้าอื่นแล้ว มันจะมีการอัปเดต session state ฉะนั้นต้องเช็คทั้งคู่ เพื่อให้ทุกเพจข้อมูลมันเชื่อมกัน  
    df = st.session_state.get('df')
    if df is None:
        st.write("Please upload a file to proceed.")
        st.markdown(""" Or you can go to 2_💻_Data Simulating if you don't have any data. """ )
    else:
        if st.button("Show Your data that's you upload, just click this button👈"):
            st.write("Your Data from upload")
            st.write(df)
            ## สร้างตัวแปรขึ้นมาเพื่อเช็คว่ามีค่าที่หายไปหรือเปล่า
            missing_variables = df.isna().sum()[lambda x: x > 0]
            ## เช็คว่ามีข้อมูลที่หายไปไหม 
            if not missing_variables.empty:
                st.write("Oops! Look like something are missing. 👻") 

if page == "Homepage":
    show()
elif page == "Data Manipulation":
    data_manipulation.show()  
elif page == "Data Simulating":
    data_simulating.show()  
elif page == "Multivariate Visualization":
    multivariate_visualization.show()
elif page == "Hypothesis Testing":
    hypothesis_testing.show()
elif page == "MANOVA":
    manova.show()
