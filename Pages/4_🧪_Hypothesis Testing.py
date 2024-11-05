import streamlit as st; import numpy as np
import pandas as pd; import pingouin as pg
from scipy.stats import chi2, f

# ส่วนแสดงผลของหน้า
st.set_page_config(
    page_title="Hypothesis Testing (1-2 groups)",
    page_icon="🧪",
)
def HT_show():
    head = """ <h1 style='text-align: center'> Hypothesis Testing in Multivariate Analysis </h1> """
    st.markdown(head, unsafe_allow_html=True)
    st.write("# ") 

    st.write("""
             **Description:**
        This page focuses on hypothesis testing related to comparing the Mean Vector (μ) of your data with the expected value 
             𝜇0 through various analytical tools. Hypothesis testing is a critical step in data analysis to ensure that your data \
             aligns with the theoretical or hypothesized expectations.
        """)
HT_show()

# ส่วนการตรวจสอบข้อมูล
## ฟังก์ชันตรวจสอบข้อมูลว่ามีใน session_state ไหม
def data_check():
    keys = ['df_final', 'samples', 'swiss']
    data = [st.session_state.get(key) for key in keys if key in st.session_state]
    if data:
        return tuple(data)
    
    st.write("It seems like we don't have any data, would you return to upload file or simulate new data.")
    return None
    
## ฟังก์ชันในการเลือกชื่อคอลัมน์ของข้อมูล
def variate_vz(all_data,alphas):
    st.write("#### Choose your variate to test hypothesis, just click below👇")
    normal_columns = all_data.columns
    
    selected_options = st.multiselect('Select at least 2 columns', normal_columns)
    return selected_options
    
## ฟังก์ชันในการหาค่าเฉลี่ย และความแปรปรวน จากชื่อคอลัมน์ของข้อมูล
def variate_pp(all_data, selected_options):
    clean_data = all_data[selected_options].dropna()
    if len(selected_options) < 2:
        st.warning("Please select at least 2 columns.")
        return None, None
    else:
        result = pg.multivariate_normality(clean_data, alpha=alphas)
        if result[2]:
            st.success("Your data has pass assumption")
            y_bar = clean_data.mean().to_numpy()
            st.write("Mean vector (y_bar) of selected columns is:")
            st.write(y_bar)

            cov_mat = clean_data.cov().to_numpy()
            st.write("Covariance Matrix (S) of selected columns is:")
            st.write(cov_mat)
            return y_bar, cov_mat
        else:
            st.warning(""" It's seem like your data from columns that has been selected don't pass assumption which, 
                   Data is sampling from population that have normal distribution """)

## ฟังก์ชันรับค่า alpha
def input_alpha():
    option = [0.01, 0.05, 0.1]
    selected_options = st.selectbox('Select your alpha', option)
    return selected_options

## ฟังก์ชันรับค่า vector μ
def input_mu_vector(n_key1):
    vec_mu = np.zeros((p))
    for i in range(p):
        vec_mu[i] = st.number_input(f'Value at ({i+1},1)', key=f"{n_key1}_{i}")
    return vec_mu

## ฟังก์ชันรับค่า covariance matrix
def input_cov_matrix(n_key2):
    cov_matrix = np.zeros((p, p))
    for i in range(p):
        cols = st.columns(p)
        for j in range(p):
            cov_matrix[i, j] = cols[j].number_input(f'Value at ({i+1},{j+1})', key=f"{n_key2}_{i}_{j}",min_value=0.0)
    return cov_matrix

## Check matrix is symmetric?
def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T) 

## Check matrix is positive semi-definite?
def is_positive_semi_definite(matrix):
    try:
        # Check if all eigenvalues are non-negative
        return np.all(np.linalg.eigvals(matrix) >= 0)
    except np.linalg.LinAlgError:
        return False

# Covariance matrix validation
def validate_cov_matrix(matrix):
    if not is_symmetric(matrix):
        st.error("The covariance matrix is not symmetric. Please enter a valid symmetric matrix.")
        return False
    if not is_positive_semi_definite(matrix):
        st.error("The covariance matrix is not positive semi-definite. Please enter a valid covariance matrix.")
        return False
    return True

# ฟังก์ชันทดสอบสมมุติฐาน
## ฟังก์ชันทดสอบแบบ 1 กลุ่ม ทราบความแปรปรวน
def Test_muemu0_with_kS(y_bar, vector_mu, cov_mat, alphas, n):
    diff = y_bar - vector_mu
    Sigma_inv = np.linalg.inv(cov_mat)
    Z_squared = n * np.matmul(np.matmul(diff, Sigma_inv), diff.T)
    cutpoint = chi2.ppf(1-alphas, len(diff))
    if Z_squared > cutpoint:
        st.warning(f"Z^2 is {Z_squared:.4f} which > cutpoint is {cutpoint:.4f}, therefore reject H0")
    else:
        st.success(f"Z^2 is {Z_squared:.4f} which <= cutpoint is {cutpoint:.4f}, therefore accept H0")

## ฟังก์ชันทดสอบแบบ 1 กลุ่ม ไม่ทราบความแปรปรวน
def Test_muemu0_with_ukS(y_bar, vector_mu, cov_mat, alphas, n):
    v = n-1
    diff = y_bar - vector_mu
    p = len(diff)
    Sigma_inv = np.linalg.inv(cov_mat)
    T_squared = n * np.matmul(np.matmul(diff, Sigma_inv), diff.T)
    cutpoint = f.ppf(1-alphas, p, v-p+1) * ((v * p) / (v-p+1))
    if np.all(vector_mu == 0):
        st.error(f'Your poppulation mu matrix is 0, Please input the correct data')
    else:
        if T_squared > cutpoint:
            st.warning(f'T^2 is {T_squared:.4f} which > cutpoint is {cutpoint:.4f}, therefore reject H0')
        else:
            st.success(f'T^2 is {T_squared:.4f} which <= cutpoint is {cutpoint:.4f}, therefore accept H0')

## ฟังก์ชันทดสอบแบบ 2 กลุ่ม ที่เป็นอิสระกัน
def Test_mu1emu2(y_bar1, y_bar2, n_1, n_2, s_1, s_2, alphas):
    diff = y_bar1 - y_bar2
    p = len(diff)
    S_pool = (1/(n_1+n_2-2)) * ((n_1-1)*s_1 + (n_2-1)*s_2)
    S_pool_inv = np.linalg.inv(S_pool)
    T_squared = ((n_1 * n_2)/(n_1 + n_2)) * np.matmul(np.matmul(diff, S_pool_inv), diff.T)
    cutpoint = f.ppf(1-alphas, p, n_1+n_2-p-1) * (((n_1+n_2-2)*p) / (n_1+n_2-p-1))
    if T_squared > cutpoint:
        st.warning(f'T^2 is {T_squared:.4f} which > cutpoint is {cutpoint:.4f}, therefore reject H0')
    else:
        st.success(f'T^2 is {T_squared:.4f} which <= cutpoint is {cutpoint:.4f}, therefore accept H0')

## ฟังก์ชันเลือกสมมุติฐาน
def Select_hypo():
    st.write("#### Choose your Hypothesis")
    options = ["Test sample with know Σ", "Test sample with unknow Σ", "Test 2 samples"]
    selected_options2 = st.selectbox('Select Hypothesis fits your data', options)
    return selected_options2

# การดำเนินการ
def check_hypo(selected_options, y_bar, cov_mat, alphas, n, p):
    if selected_options == "Test sample with know Σ":
        st.markdown("<h3 style='text-align: center; margin-top: 50px; margin-bottom: 5px;'>Test sample with known Σ</h3>", unsafe_allow_html=True)
        st.latex(r"H_0: \mu = \mu_0 \quad \text{vs.} \quad H_1: \mu \neq \mu_0")
        st.write("#### Input mean vector (μ) of population")
        vec_mu = input_mu_vector("mu0")
        st.write("#### Input covariance matrix (Σ) of population")
        cov_mat = input_cov_matrix("B_Sigma")
        try:
            if validate_cov_matrix(cov_mat):  
                st.write("Mean Vector (μ) of population is:")
                st.write(vec_mu)
                st.write("Covariance matrix (Σ) of population is:")
                st.write(cov_mat)
                Test_muemu0_with_kS(y_bar, vec_mu, cov_mat, alphas, n)
            else:
                st.write("Covariance matrix is not symmetric or positive semidefinite matrix.")  # แสดงข้อความข้อผิดพลาดที่เหมาะสม
        except:
            st.write("Something wrong with your Covariance Matrix  Please Check again")  # แสดงข้อความข้อผิดพลาดที่เหมาะสม
        
    elif selected_options == "Test sample with unknow Σ":
        st.markdown("<h3 style='text-align: center; margin-top: 50px; margin-bottom: 5px;'>Test sample with unknow Σ</h3>", unsafe_allow_html=True)
        st.latex(r"H_0: \mu = \mu_0 \quad \text{vs.} \quad H_1: \mu \neq \mu_0")
        st.write("#### Input mean vector (μ) of population")
        vec_mu = input_mu_vector("mu0")
        st.write("Mean Vector (μ) of population is:")
        st.write(vec_mu)
        st.write("Covariance matrix (S) of sample is:")
        st.write(cov_mat)
        if validate_cov_matrix(cov_mat):
            Test_muemu0_with_ukS(y_bar, vec_mu, cov_mat, alphas, n)
        else: 
            st.write("Something wrong with your matrix")
        
    elif selected_options == "Test 2 samples":
        st.markdown("<h3 style='text-align: center; margin-top: 50px; margin-bottom: 5px;'>Test 2 samples</h3>", unsafe_allow_html=True)
        st.latex(r"H_0: \mu_1 = \mu_2 \quad \text{vs.} \quad H_1: \mu_1 \neq \mu_2")
        n_1 = n
        options = all_data.columns.tolist()
        selected_options2 = st.multiselect('Select at least 2 columns', options, key='multiselect2')
        if selected_options2:
            y_bar2, cov_mat2 = variate_pp(all_data, selected_options2) # เอาคอลัมน์ที่เลือกไปหาค่าเฉลี่ย และส่วนเบี่ยงเบนมาตรฐาน 
            if y_bar2 is not None and cov_mat2 is not None:
                y_bar1 = y_bar
                cov_mat1 = cov_mat
                n_2 = n  
                Test_mu1emu2(y_bar1, y_bar2, n_1, n_2, cov_mat1, cov_mat2, alphas)
        else:
            y_bar1 = y_bar
            S_mat1 = cov_mat
            st.write("#### Input your mean vector 2 (y_bar2)")
            y_bar2 = input_mu_vector("y_bar2")
            st.write("#### Input your Covariace matrix 2 (S2)")
            S_mat2 = input_cov_matrix("S_mat2")
            n_2 = n  
            Test_mu1emu2(y_bar1, y_bar2, n_1, n_2, S_mat1, S_mat2, alphas)    

data = data_check() # ข้อมูลทั้งหมด
st.write('Choose your input method')
check_1 = st.checkbox('Your data from uploading.', key='input_data_checkbox_1')
if check_1:
    if data is not None:
        st.write("Your data from previous page is:")
        if isinstance(data, tuple):  # กรณีที่ data เป็น tuple ซึ่งมีทั้ง df_final และ samples
            # แปลงข้อมูลที่ได้เป็น DataFrame
            dataframes = [pd.DataFrame(item) if isinstance(item, (list, pd.Series)) else item for item in data]
            
            # รวม DataFrames
            all_data = pd.concat(dataframes, axis=1)
            
            # ปรับชื่อคอลัมน์ที่ซ้ำกัน
            all_data.columns = pd.Series(all_data.columns).where(~all_data.columns.duplicated(), all_data.columns + '_dup')
            
            cb_alldata = st.checkbox("Show your data",key='cb_alldata')
            if cb_alldata:
                st.write(all_data)
                
        else:
            all_data = pd.DataFrame(data)
        n = len(all_data)
        alphas = input_alpha()

        selected_options = variate_vz(all_data,alphas) # เอาชื่อคอลัมน์มาเป็น ตัวเลือก
        if selected_options:
            p = len(selected_options)
            result = variate_pp(all_data, selected_options)

            if result is not None:  # ตรวจสอบว่าไม่ใช่ None ก่อน
                y_bar, cov_mat = result  # unpack ค่าเฉพาะเมื่อ result ไม่ใช่ None
                st.markdown('---')  # สร้างเส้นแบ่งแนวนอน
                selected_options2 = Select_hypo()
                check_hypo(selected_options2, y_bar, cov_mat, alphas, n, p)
            else:
                st.error("Please check your selected columns or data assumption.")
    st.markdown('---')  # สร้างเส้นแบ่งแนวนอน

check_2 = st.checkbox('Descriptive statistic but do not have data.', key='input_data_checkbox_2')
if check_2:
    st.markdown("#### Even you don't have data ")
    st.write("but you have descriptive statistics. You can also use the hypothesis testing.")
    alphas = input_alpha()
    p = st.number_input(f'number of dimension is',min_value=2)
    n = st.number_input(f'number of sample is',min_value=10,max_value=10000)
    st.write("#### Input your sample's mean vector (y_bar)")
    y_bar = input_mu_vector("y_bar")
    st.write("#### Input your sample's Covariace matrix (S)")
    S_mat = input_cov_matrix("S_mat")
    if np.all(y_bar == 0) or np.all(S_mat == 0) or not validate_cov_matrix(S_mat):
        st.write("y_bar or S is vector 0 or S Matrix isn't symmetric")
    else:
        selected_options2 = Select_hypo()
        all_data = pd.DataFrame()
        check_hypo(selected_options2, y_bar, S_mat, alphas, n, p)