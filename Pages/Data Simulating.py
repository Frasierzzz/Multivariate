def show():
    import streamlit as st; import numpy as np; import pandas as pd
    from scipy.stats import multivariate_normal; np.random.seed(0)
    # ส่วนแสดงผลของหน้า
    st.set_page_config(page_title="Data Simulating", page_icon="💻",)
    def DS_show():
        # ชื่อหัวข้อ
        head = """ <h1 style='text-align: center'> Data Simulating </h1> """
        st.markdown(head, unsafe_allow_html=True)
        st.write("# ") 
        st.write( """
                 **Description:**
            This page is designed for simulating data with numpy library. It involve simulating data
                    when you dont'have data but you already know Mean Vector (μ) or Covariance Matrix (Σ) for analysis or testing hypothesis.
                    The processes that can be performed include simulating data from μ and Σ.
            """ )
    DS_show()
    
    if 'df_final' in st.session_state:
        df_final = st.session_state.df_final
        st.write("Your data from previous page is:")
    
        cb_data = st.checkbox("Show your data",key='cb_data')
        if cb_data:
            st.write(df_final)
            if st.button("Reset Result, just click this button👈"):
                del st.session_state['df_final']
                st.write("Your data has been clear!")
    
        if 'swiss' in st.session_state:
            swiss = st.session_state.swiss
            cb_swiss = st.checkbox("Show swiss dataset",key='cb_swiss')
            if cb_swiss:
                st.write(swiss)
    
    elif 'swiss' in st.session_state:
        swiss = st.session_state.swiss
        cb_swiss = st.checkbox("Show swiss dataset",key='cb_swiss')
        if cb_swiss:
            st.write(swiss)
    
    else:
        st.write("""It seems like we don't have any data, would you return to upload file or simulate new data.
                 
                 Note: If you want to upload, just back to the homepage as follows:
                 """)
        st.write('<a href="/" target="_self">Return to Homepage</a>', unsafe_allow_html=True)
    
    # ส่วนแสดงผลของฟังก์ชันทั้งหมด
    ## Input Part
    ### Input Mean vector function
    def input_muvector(dm):
        muvector = np.zeros((dm))
        for i in range(dm):
            muvector[i] = st.number_input(f'Value at ({i+1},0)')
        return muvector
    
    ### Input Covariance Matrix function
    def input_Cov_matrix(dm):
        matrix = np.zeros((dm,dm))
        for i in range(dm):
            cols = st.columns(dm)
            for j in range(dm):
                matrix[i, j] = cols[j].number_input(f'Value at ({i+1},{j+1})', value=0.0, min_value=0.0)
        return matrix     
    
    ## Check Covariance
    def is_symmetric(matrix):
        return np.allclose(matrix, matrix.T) and np.all(np.linalg.eigvals(matrix) >= 0)  
    
    def my_Simulate_Sample(vec_mu,cov_mat,n):
        samples = multivariate_normal.rvs(vec_mu,cov_mat,n)
        return samples
    
    if 'samples' in st.session_state:
        samples = st.session_state.samples
        cb_samples = st.checkbox("Show your samples",key='cb_sample')
        if cb_samples:
            st.write(samples.T)
            if st.button("Reset Samples"):
                del st.session_state['samples']
                st.write("Samples have been reset.")
    
    st.markdown('---')  # สร้างเส้นแบ่งแนวนอน
    
    # ส่วนเนื้อหา
    st.write( "#### Or if you want to simulate new data, just click below👇" )
    dm = st.number_input("Enter your size of vector:", min_value=1, max_value=10, step=1)
    
    if dm != 0:
        st.write("### Input population mean vector (μ):")
        mu_vector = input_muvector(dm)
        st.write(f"### Input population covariance matrix (σ):")
        cov_mat = input_Cov_matrix(dm)
            
        st.write("### Simulating samples")
        n = st.number_input("Enter number of samples :", min_value=10, max_value=10000,value=10)
        st.write("Min: 10, Max: 10000")
        vec_mu = mu_vector.T # Transpose Vector mu
        cb_2 = st.checkbox("Show samples.")
        if cb_2:
            try:
                if is_symmetric(cov_mat):    
                    samples = my_Simulate_Sample(vec_mu,cov_mat,n)
                    if 'samples' in st.session_state:
                        existing_cols = st.session_state.samples.shape[1]
                    else:
                        existing_cols = 0
                    samples = pd.DataFrame(samples, columns=[f'Sim_{existing_cols + i+1}' for i in range(dm)])
                    st.write(samples.T)
                    if st.button("Save Result, just click this button👈"):
                        if 'samples' in st.session_state:
                            st.session_state.samples = pd.concat([st.session_state.samples, samples], axis=1)
                        else:
                            st.session_state.samples = samples
                            st.write("Your samples has been save!")
                else:
                    st.write("Covariance matrix is not symmetric or positive semidefinite matrix.")  # แสดงข้อความข้อผิดพลาดที่เหมาะสม
            except:
                st.write("Something wrong with your Covariance Matrix  Please Check again")  # แสดงข้อความข้อผิดพลาดที่เหมาะสม
    

