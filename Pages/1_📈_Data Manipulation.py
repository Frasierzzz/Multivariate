import streamlit as st; import math

# ส่วนแสดงผลของหน้า
st.set_page_config(
    page_title="Data Manipulation", page_icon="📈",)
## ฟังก์ชันแสดงข้อความบนหน้าเว็บ
def DM_show():
    # ชื่อหัวข้อ
    head = """ <h1 style='text-align: center'> Data Manipulation </h1> """
    st.markdown(head, unsafe_allow_html=True)
    st.write("# ") 

    st.write( """
             **Description:**
        This page is created for data preparation with pandas library. It involves cleaning the original 
                data to make it ready for further analysis. The processes that can be performed include data 
                filtering (filtering as nume- ric data), dropping columns with more than 50% missing data and filling 
                missing values in some columns with the mean value. 

                Note: These mentioned processes are only applicable to quantitative data.
        """ )
DM_show()

# ส่วนตรวจสอบไฟล์
## ฟังก์ชันแสดงว่ามีการดรอปคอลัมน์ไหนบ้าง
def col_drop(dropped_columns):
    for col in dropped_columns:
        st.write(f' " {col} "  column has been dropped')

## เช็คว่า df มันมีค่าใน session_state ไหมถ้าไม่มี จะไล่ให้กลับไปอัพไฟล์ หรือไม่ก็ข้ามส่วนนี้ไปเลย
if 'df_final' in st.session_state:
    df_final = st.session_state.df_final

    st.write("Your data from previous page is:")
    cb_data = st.checkbox("Show your data",key='cb_data')
    if cb_data:
        st.write(df_final)
        if st.button("Reset Result, just click this button👈"):
            del st.session_state['df_final']
            st.write("Your data has been clear!")

elif 'df' in st.session_state:
    df = st.session_state.df
    df_final = df

    ### สร้างตัวแปรขึ้นมาเพื่อเช็คว่ามีค่าที่หายไปหรือเปล่า
    missing_variables = df.isna().sum()[lambda x: x > 0]
    ### เช็คว่ามีข้อมูลที่หายไปไหม 
    if not missing_variables.empty:
        st.write("Oops! Look like something are missing. 👻")

    ### กล่องแรก แสดงว่ามีตัวแปรใดบ้างที่มีค่า missing
    cb_1 = st.checkbox("Show Missing Valuable")
    if cb_1:
        st.write(missing_variables)

    ### การเตรียมข้อมูล
    st.write("### Data Preparation")
    #### กล่องสอง ให้โชว์แค่ที่เป็นตัวเลข
    cb_2 = st.checkbox("Show numeric Data only")
    if cb_2:
        df_scale = df.select_dtypes(include=['number'])
        dropped_columns_1 = set(df_final.columns) - set(df_scale) #### ต้องการทราบคอลัมน์ที่ดรอปไป
        col_drop(dropped_columns_1)
        df_final = df_scale

    #### กล่องสาม ให้ตัดคอลัมน์ที่มีจำนวนข้อมูล < thresh
    cb_3 = st.checkbox("Drop Valuable that's missing > 50%")
    if cb_3:
        df_fill_1 = df_final.copy() # ก๊อปปี้มาเพื่อกันการเกิดปัญหา
        df_drop = df_fill_1.dropna(thresh=round(len(df)/2), axis=1) # เอาขนาดของข้อมูลมาหาร 2 แล้วปัดขึ้น
        dropped_columns_2 = set(df_fill_1.columns) - set(df_drop.columns)  #### ต้องการทราบคอลัมน์ที่ดรอปไป
        col_drop(dropped_columns_2)
        df_final = df_drop

    #### กล่องสี่ ให้เช็คว่าคอลัมน์ไหนที่ข้อมูลหายจะเติมเพิ่มด้วยค่าเฉลี่ย
    cb_4 = st.checkbox("Fill Missing Values with Mean (Numeric columns only)")
    if cb_4:
        number_cols = df.select_dtypes(include=['int','float']).columns # เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข
        df_fill = df_final.copy() # ก๊อปปี้มาเพื่อกันการเกิดปัญหา
        for col in number_cols: # วนตัวลูปในตัวแปร คอลัมน์ เพื่อเติมคอลัมน์ที่ไม่ครบ
            if df[col].isnull().any(): # เช็คว่าคอลัมน์ไหนที่มีค่าหายไป
                df_fill[col] = df[col].fillna(math.ceil(df[col].mean())) # เติมคอลัมน์นั้นด้วยค่าเฉลี่ย
        df_final = df_fill

    st.write("Final Data")
    st.write(df_final)
    if st.button("Save Result, just click this button👈"):
        st.session_state['df_final'] = df_final
        st.write("Your data has been save!")
## สืบเนื่องมาจากหน้าแรก ถ้าไม่ยอมอัพโหลดไฟล์ ก็กลับไปอัพซะ
else:
    st.write("Oops! Look like you forgot to upload your file")
    st.write('<a href="/" target="_self">Return to Homepage</a>', unsafe_allow_html=True)

    

