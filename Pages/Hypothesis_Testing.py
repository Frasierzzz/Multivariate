def show():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import pingouin as pg
    from scipy.stats import chi2, f
    
    # ส่วนแสดงผลของหน้า
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
    def variate_vz(all_data, alphas):
        st.write("#### Choose your variate to test hypothesis, just click below👇")
        normal_columns = all_data.columns
        selected_options = st.multiselect('Select at least 2 columns', normal_columns, key='multiselect_variate')
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
                st.success("Your data has passed the assumption check.")
                y_bar = clean_data.mean().to_numpy()
                st.write("Mean vector (y_bar) of selected columns is:")
                st.write(y_bar)
    
                cov_mat = clean_data.cov().to_numpy()
                st.write("Covariance Matrix (S) of selected columns is:")
                st.write(cov_mat)
                return y_bar, cov_mat
            else:
                st.warning(""" Your selected columns do not pass the assumption. 
                       Ensure that data is sampled from a normally distributed population. """)
    
    ## ฟังก์ชันรับค่า alpha พร้อมกำหนด key
    def input_alpha(key='alpha_select'):
        option = [0.01, 0.05, 0.1]
        selected_options = st.selectbox('Select your alpha', option, key=key)
        return selected_options
    
    ## ฟังก์ชันรับค่า vector μ พร้อม key
    def input_mu_vector(n_key1, p):
        vec_mu = np.zeros((p))
        for i in range(p):
            vec_mu[i] = st.number_input(f'Value at ({i+1},1)', key=f"{n_key1}_{i}")
        return vec_mu
    
    ## ฟังก์ชันรับค่า covariance matrix พร้อม key
    def input_cov_matrix(n_key2, p):
        cov_matrix = np.zeros((p, p))
        for i in range(p):
            cols = st.columns(p)
            for j in range(p):
                cov_matrix[i, j] = cols[j].number_input(f'Value at ({i+1},{j+1})', key=f"{n_key2}_{i}_{j}", min_value=0.0)
        return cov_matrix
    
    # ฟังก์ชันทดสอบสมมุติฐาน
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
            st.error('Population mean vector (μ) is 0. Please input the correct data')
        else:
            if T_squared > cutpoint:
                st.warning(f'T^2 is {T_squared:.4f} which > cutpoint is {cutpoint:.4f}, therefore reject H0')
            else:
                st.success(f'T^2 is {T_squared:.4f} which <= cutpoint is {cutpoint:.4f}, therefore accept H0')
    
    ## ฟังก์ชันเลือกสมมุติฐานพร้อม key
    def Select_hypo(key='hypo_select'):
        st.write("#### Choose your Hypothesis")
        options = ["Test sample with known Σ", "Test sample with unknown Σ", "Test 2 samples"]
        selected_options2 = st.selectbox('Select Hypothesis fits your data', options, key=key)
        return selected_options2
    
    # เรียกใช้ฟังก์ชันต่าง ๆ
    data = data_check()
    st.write('Choose your input method')
    check_1 = st.checkbox('Your data from uploading.', key='input_data_checkbox_1')
    if check_1:
        if data is not None:
            st.write("Your data from previous page is:")
            if isinstance(data, tuple):
                dataframes = [pd.DataFrame(item) if isinstance(item, (list, pd.Series)) else item for item in data]
                all_data = pd.concat(dataframes, axis=1)
                all_data.columns = pd.Series(all_data.columns).where(~all_data.columns.duplicated(), all_data.columns + '_dup')
                
                cb_alldata = st.checkbox("Show your data", key='cb_alldata')
                if cb_alldata:
                    st.write(all_data)
                    
            else:
                all_data = pd.DataFrame(data)
            n = len(all_data)
            alphas = input_alpha(key='alpha_select_1')
    
            selected_options = variate_vz(all_data, alphas)
            if selected_options:
                p = len(selected_options)
                result = variate_pp(all_data, selected_options)
    
                if result is not None:
                    y_bar, cov_mat = result
                    st.markdown('---')
                    selected_options2 = Select_hypo(key='hypo_select_1')
                    check_hypo(selected_options2, y_bar, cov_mat, alphas, n, p)
                else:
                    st.error("Please check your selected columns or data assumption.")
        st.markdown('---')
    
    check_2 = st.checkbox('Descriptive statistic but do not have data.', key='input_data_checkbox_2')
    if check_2:
        st.markdown("#### Even you don't have data ")
        st.write("but you have descriptive statistics. You can also use the hypothesis testing.")
        alphas = input_alpha(key='alpha_select_2')
        p = st.number_input(f'number of dimension is', min_value=2, key='dim_input')
        n = st.number_input(f'number of sample is', min_value=10, max_value=10000, key='sample_input')
        st.write("#### Input your sample's mean vector (y_bar)")
        y_bar = input_mu_vector("y_bar_input", p)
        st.write("#### Input your sample's Covariance matrix (S)")
        S_mat = input_cov_matrix("S_mat_input", p)
        if np.all(y_bar == 0) or np.all(S_mat == 0) or not validate_cov_matrix(S_mat):
            st.write("y_bar or S is vector 0 or S Matrix isn't symmetric")
        else:
            selected_options2 = Select_hypo(key='hypo_select_2')
            check_hypo(selected_options2, y_bar, S_mat, alphas, n, p)
