def show():
    import streamlit as st; import numpy as np
    import matplotlib.pyplot as plt; import pandas as pd
    from scipy.stats import multivariate_normal
    import plotly.graph_objects as go
    # ส่วนแสดงผลของหน้า
    st.set_page_config(
        page_title="Multivariate Visualization",
        page_icon="📊",
    )
    def MV_show():
        head = """ <h1 style='text-align: center'> Data Visualization </h1> """
        st.markdown(head, unsafe_allow_html=True)
        st.write("# ") 
        st.write("""
                 **Description:**
                 This page is designed for visualizing bivariate distributions by using the NumPy and Matplotlib libraries. 
                 These visualizations are particularly useful when analyzing or testing hypotheses involving bivariate data. 
                 The tools provided allow you to visualize data based on the Mean Vector (μ) and Covariance Matrix (Σ).
                 """)
    MV_show()
    
    # ส่วนแสดงผลของฟังก์ชันทั้งหมด 
    ## Distribution Function
    def my_plot_mvn(pdf, X, Y, vec_mu, cov_mat, gtitle):
        std_dev_x = np.sqrt(cov_mat[0, 0])
        std_dev_y = np.sqrt(cov_mat[1, 1])
        fig_1 = go.Figure(data=[go.Surface(z=pdf, x=X, y=Y)])
        
        # Define limits based on the mean vector and std deviations
        x_min = vec_mu[0] - 3 * std_dev_x
        x_max = vec_mu[0] + 3 * std_dev_x
        y_min = vec_mu[1] - 3 * std_dev_y
        y_max = vec_mu[1] + 3 * std_dev_y
        
        fig_1.update_layout(title=gtitle, autosize=False, width=500, height=500,
                            margin=dict(l=65, r=50, b=65, t=90),
                            scene=dict(xaxis=dict(range=[x_min, x_max]),
                                       yaxis=dict(range=[y_min, y_max]),
                                       zaxis=dict(range=[0, np.max(pdf)])))
        st.plotly_chart(fig_1)
    
    ## Contour Function
    def my_plot_contour(pdf, X, Y, vec_mu, cov_mat):
        fig_2, ax = plt.subplots()
        contour = ax.contour(X, Y, pdf)
        ax.set_title('Contour')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
        # Define limits based on the mean vector and std deviations
        std_dev_x = np.sqrt(cov_mat[0, 0])
        std_dev_y = np.sqrt(cov_mat[1, 1])
        ax.set_xlim([vec_mu[0] - 3 * std_dev_x, vec_mu[0] + 3 * std_dev_x])
        ax.set_ylim([vec_mu[1] - 3 * std_dev_y, vec_mu[1] + 3 * std_dev_y])
        
        st.pyplot(fig_2)
    
    ## Check Data
    def data_check():
        keys = ['df_final', 'samples', 'swiss']
        data = [st.session_state.get(key) for key in keys if key in st.session_state]
        if data:
            return data
        
        st.write("It seems like we don't have any data, would you return to upload file or simulate new data.")
        return None
    
    def variate_vz(all_data):
        st.write("#### Choose your variate to do visualization, just click below👇")
        options = all_data.columns.tolist()
        selected_options = st.multiselect('Select exactly 2 columns', options)
        return selected_options
    
    def variate_pp(selected_options, all_data):
        if len(selected_options) != 2:
            st.warning("Please select exactly 2 columns.")
        else:
            vec_mu = all_data[selected_options].dropna().mean().to_numpy()
            st.write("Mean vector of Selected columns is:")
            st.write(vec_mu)
    
            cov_mat = all_data[selected_options].dropna().cov().to_numpy()
            st.write("Covariance Matrix of Selected columns is:")
            st.write(cov_mat)
    
            binormal = multivariate_normal(vec_mu, cov_mat)
            
            # Define bounds based on data
            X = np.linspace(all_data[selected_options[0]].min() - 1, all_data[selected_options[0]].max() + 1, 200)
            Y = np.linspace(all_data[selected_options[1]].min() - 1, all_data[selected_options[1]].max() + 1, 200)
            X, Y = np.meshgrid(X, Y)
            pos = np.dstack([X, Y])
            pdf = binormal.pdf(pos)
    
            cb_11 = st.checkbox("Show --> Bivariate normal distribution")
            if cb_11:
                my_plot_mvn(pdf, X, Y, vec_mu, cov_mat, "Bivariate normal distribution")
                st.markdown('---')  # สร้างเส้นแบ่งแนวนอน
            
            cb_21 = st.checkbox("Show --> Contour")
            if cb_21:
                my_plot_contour(pdf, X, Y, vec_mu, cov_mat)
    
    # ส่วนเนื้อหา
    data = data_check()
    if data is not None:
        st.write("Your data from previous page is:")
        
        # แปลงข้อมูลที่ได้เป็น DataFrame
        dataframes = [pd.DataFrame(item) if isinstance(item, (list, pd.Series)) else item for item in data] # เช็คว่าเป็น list หรือ pd ไหม ถ้าเป็น
        # จะแปลงค่าเป็น dataframe ถ้าไม่ก็คืนค่าปกติ
        
        # รวม DataFrames
        all_data = pd.concat(dataframes, axis=1)
        
        # ปรับชื่อคอลัมน์ที่ซ้ำกัน
        all_data.columns = pd.Series(all_data.columns).where(~all_data.columns.duplicated(), all_data.columns + '_dup')
        cb_alldata = st.checkbox("Show your data",key='cb_alldata')
        if cb_alldata:
            st.write(all_data)
    
        selected_options = variate_vz(all_data)
        variate_pp(selected_options, all_data)
