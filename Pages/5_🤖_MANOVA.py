def show():
    import streamlit as st; import pandas as pd
    import numpy as np; import pingouin as pg
    from statsmodels.multivariate.manova import MANOVA
    
    # ส่วนแสดงผลของหน้า
    st.set_page_config(
        page_title="Hypothesis Testing (>2 groups)",
        page_icon="🤖",
    )
    
    def MANOVA_show():
        st.markdown("<h1 style='text-align: center'> MANOVA Testing </h1>", unsafe_allow_html=True)
        st.write("## ")
        st.write("""
                 **Description:**
            This page is dedicated to conducting MANOVA (Multivariate Analysis of Variance) tests. 
            Ensure that your data is appropriate for MANOVA, meaning:
            - All independent variables follow a normal distribution.
            - Covariance matrices across groups are homogeneous.
            """)
    
    MANOVA_show()
    
    def data_check():
        keys = ['df_final', 'samples', 'swiss']
        data = [st.session_state.get(key) for key in keys if key in st.session_state]
        if data:
            combined_data = pd.concat([pd.DataFrame(item) for item in data], axis=1)
            combined_data.columns = pd.Series(combined_data.columns).where(~combined_data.columns.duplicated(), combined_data.columns + '_dup')
            return combined_data
        
        st.write("It seems like we don't have any data, would you return to upload file or simulate new data.")
        return None
    
    def input_alpha():
        option = [0.01, 0.05, 0.1]
        selected_options = st.selectbox('Select your alpha', option)
        return selected_options
    
    def multi_normality_test(groups_data):
        results = [pg.multivariate_normality(groups_data[f'Group {i+1}'], alpha=alphas) for i in range(num_groups)]
        boolean_results = [result[2] for result in results]
        p_values = [result[1] for result in results]
        group_names = [f'Group {i+1}' for i in range(num_groups)]
        failed_groups = [group_names[i] for i, passed in enumerate(boolean_results) if not passed]
        for i in range(num_groups):
            st.write(f"{group_names[i]}: p-value = {p_values[i]:.4f}, {'Reject' if p_values[i] < alphas else 'Fail to reject'} the null hypothesis: This Group is randomly selected from a normally distributed population. ")
        return boolean_results, p_values, failed_groups
    
    def create_multiselectboxes(normal_columns, num_groups):
        selected_columns = {}
        for i in range(num_groups):
            selected_columns[f'Group {i+1}'] = st.multiselect(f'Select at least 2 variables for Group {i+1}', normal_columns, key=f'select_{i+1}')
        
        selected_values = [item for sublist in selected_columns.values() for item in sublist]
        if len(set(selected_values)) != len(selected_values):
            st.error("Duplicate variables selected across groups. Please choose different variables for each group.")
            return None
        return selected_columns
    
    def validate_selection(selected_columns):
        for group, columns in selected_columns.items():
            if len(columns) < 2:
                st.error(f"{group} has less than 2 variables. Please select at least 2 variables.")
                return False
        return True
    
    def box_m_test_pingouin(groups_data):
        combined_data = pd.concat([pd.DataFrame(data) for data in groups_data.values()])
        group_labels = np.concatenate([[group] * len(groups_data[group]) for group in groups_data])
        combined_data['group'] = group_labels
        dvs = combined_data.columns[:-1]  # เลือกทุกคอลัมน์ยกเว้นคอลัมน์กลุ่ม
    
        box_m_result = pg.box_m(data=combined_data, dvs=dvs.tolist(), group='group')
        return box_m_result
    
    def perform_MANOVA(groups_data):
        combined_data = pd.concat([pd.DataFrame(groups_data[group], columns=[f'Var_{i}' for i in range(groups_data[group].shape[1])]) for group in groups_data], ignore_index=True)
        group_labels = np.concatenate([[group] * len(groups_data[group]) for group in groups_data])
    
        manova_data = pd.DataFrame(combined_data)
        manova_data['group'] = group_labels
        maov = MANOVA.from_formula(' + '.join([f'Var_{i}' for i in range(groups_data[list(groups_data.keys())[0]].shape[1])]) + ' ~ group', data=manova_data)
    
        manova_results = maov.mv_test()
        
        # ดึงเฉพาะส่วนของผลลัพธ์ที่ต้องการแปลงเป็น DataFrame
        test_summary = manova_results['group']['stat'].T  # Transpose เพื่อให้แถวกลายเป็นคอลัมน์
    
        # แปลงเป็น DataFrame
        results_df = pd.DataFrame(test_summary)
        
        st.write("MANOVA Results Table:")
        st.dataframe(results_df)  # ใช้ st.dataframe ใน Streamlit เพื่อแสดงผลลัพธ์ในรูปแบบตาราง
        
        return results_df
    
    # def display_mean_sd(groups_data):
    #     for group_name, data in groups_data.items():
    #         group_df = pd.DataFrame(data, columns=[f'Var_{i}' for i in range(data.shape[1])])
    #         means = group_df.mean()
    #         sds = group_df.std()
            
    #         st.write(f"Statistics for {group_name}:")
    #         st.write("Mean values:")
    #         st.write(means)
    #         st.write("Standard deviation values:")
    #         st.write(sds)
    
    st.latex(r"H_0: \begin{pmatrix} \mu_{11} \\ \mu_{12} \\ \vdots \\ \mu_{1p} \end{pmatrix} = \begin{pmatrix} \mu_{21} \\ \mu_{22} \\ \vdots \\ \mu_{2p} \end{pmatrix} = \cdots = \begin{pmatrix} \mu_{k1} \\ \mu_{k2} \\ \vdots \\ \mu_{kp} \end{pmatrix}")
    
    alphas = input_alpha()
    all_data = data_check()
    
    if all_data is not None:
        st.write("Your data from previous page is:")
        st.write(all_data)
    
        columns = all_data.columns    
        num_groups = st.number_input('Select number of groups (3-5)', min_value=3, max_value=5)
        selected_columns = create_multiselectboxes(columns, num_groups)
        
        if selected_columns and validate_selection(selected_columns):
            groups_data = {group: all_data[columns].dropna().values for group, columns in selected_columns.items()}
            results, p_values, failed_groups = multi_normality_test(groups_data)   
    
            if all(results):
                # คำนวณค่า Box M test
                box_m_result = box_m_test_pingouin(groups_data)
                # display_mean_sd(groups_data)
                chi2_value = box_m_result['Chi2'][0]
                pval_value = box_m_result['pval'][0]
                # แสดงผลลัพธ์ของ Box M Test
                st.write("Box M Test Results:")
                st.write(box_m_result)
                # if pval_value >= alphas:
                #     st.write(f" p-value = {pval_value:.4f} which >= {alphas}. Therefore, Fail to reject the null hypothesis: Covariance matrices are equal across the groups.")
                    # ทดสอบ MANOVA
                manova_results = perform_MANOVA(groups_data)
                wilks_p_value = manova_results['Wilks\' lambda']['Pr > F']
    
                if wilks_p_value >= alphas:
                    st.write(f"p-value = {wilks_p_value:.4f} which >= {alphas}, Therefore, Fail to reject the null hypothesis: The means of all groups are not significantly different.")
                else:
                    st.write(f"p-value = {wilks_p_value:.4f} which < {alphas}, Therefore, we reject the null hypothesis: The means of all groups are significantly different.")
                st.warning(""" However, Test statistic that we show it's just only Wilk's lambda that can use when data has passs all assumtions and size of sample per group are equal 
                           if your data has more specific format. You can use another Test statistic that we show you from the table. """)
    
                # else:
                #     st.write(f" p-value = {pval_value:.4f} which < {alphas}. Therefore, Reject the null hypothesis: Covariance matrices are not equal across the groups.")
            else:
                st.write(f"It seems that the following data groups did not meet the required conditions.")
    else:
        st.warning("No data found in session_state. Please go back to the previous page and upload data.")



