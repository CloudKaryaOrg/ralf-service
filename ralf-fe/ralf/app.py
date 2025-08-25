import streamlit as st
import pandas as pd
from ralf import get_system_info
from ralf import Ralf
import os
import json

st.set_page_config(
    page_title="RALF - Resource-Aware LLM Finetuning",
    page_icon="🤖",
    layout="wide"
)

# ---------------------- STYLES ----------------------
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .success-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #DFF2BF;
        color: #4F8A10;
    }
    .error-message {
        padding: 1rem;
        border-radius: 5px;
        background-color: #FFE8E6;
        color: #D8000C;
    }
    .dataset-table th {
        text-align: center !important;
        font-weight: bold !important;
        padding: 12px 8px;
        border: 1px solid #ddd;
    }
    .dataset-table td {
        text-align: left;
        padding: 8px;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------- MAIN ----------------------
def main():
    st.title("🤖 RALF - Resource-Aware LLM Finetuning")

    # Sidebar
    with st.sidebar:
        st.markdown("## 🖥️ System Information")
        sys_info = get_system_info()
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="GPU", value=sys_info["GPU Available"])
            st.metric(label="GPU Memory", value=sys_info["GPU Memory"])
        with col2:
            st.metric(label="GPU Model", value=sys_info["GPU Model"])
            st.metric(label="System RAM", value=sys_info["System RAM"])

        st.markdown("---")
        st.header("API Configuration")
        hf_token = st.text_input("Hugging Face Token", type="password")
        api_choice = st.radio("Choose API for Recommendations", ["OpenAI", "Gemini"])
        if api_choice == "OpenAI":
            openai_key = st.text_input("OpenAI API Key", type="password")
            gemini_key = None
        else:
            gemini_key = st.text_input("Gemini API Key", type="password")
            openai_key = None

        if st.button("Validate Credentials"):
            if not hf_token:
                st.error("Please provide a Hugging Face token")
            elif api_choice == "OpenAI" and not openai_key:
                st.error("Please provide an OpenAI API key")
            elif api_choice == "Gemini" and not gemini_key:
                st.error("Please provide a Gemini API key")
            else:
                try:
                    ralf = Ralf(HF_TOKEN=hf_token,
                                OPENAI_API_KEY=openai_key,
                                GEMINI_API_KEY=gemini_key)
                    st.success("Credentials validated successfully!")
                    st.session_state['credentials_valid'] = True
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")
                    st.session_state['credentials_valid'] = False

        st.markdown("---")
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if df.empty:
                    st.error("The uploaded CSV file is empty")
                else:
                    st.write(f"File size: {uploaded_file.size/1024:.2f} KB")
                    st.write(f"Number of rows: {len(df)}")
                    st.write(f"Number of columns: {len(df.columns)}")
                    st.session_state['df'] = df
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")

    # Tabs
    tabs = st.tabs(["Analysis", "Recommendation", "Augmentation", "Lustration", "Futureproofing"])

    # ---------------------- ANALYSIS TAB ----------------------
    with tabs[0]:
        st.header("Dataset Analysis")
        if 'df' in st.session_state:
            st.subheader("Data Preview")

            col1, col2 = st.columns(2)
            with col1:
                st.session_state['source_col'] = st.selectbox(
                    "Select source column",
                    st.session_state['df'].columns
                )
            with col2:
                st.session_state['target_col'] = st.selectbox(
                    "Select target column",
                    [col for col in st.session_state['df'].columns if col != st.session_state['source_col']]
                )
            st.markdown(st.session_state['df'].head().to_html(escape=False, index=False, classes='dataset-table'), unsafe_allow_html=True)

            # Data cleaning (moved from Lustration tab)
            st.subheader("Data Cleaning")
            remove_nulls = st.checkbox("Remove rows with missing values")
            lowercase_text = st.checkbox("Convert text to lowercase")
            remove_duplicates = st.checkbox("Remove duplicate rows")

            if st.button("Clean Data"):
                try:
                    cleaned_df = st.session_state['df'].copy()
                    if remove_nulls:
                        cleaned_df.dropna(inplace=True)
                    if lowercase_text:
                        cleaned_df[st.session_state['source_col']] = cleaned_df[st.session_state['source_col']].str.lower()
                        cleaned_df[st.session_state['target_col']] = cleaned_df[st.session_state['target_col']].str.lower()
                    if remove_duplicates:
                        cleaned_df.drop_duplicates(inplace=True)
                    st.session_state['df'] = cleaned_df
                    st.success("Data cleaned successfully!")
                    st.markdown(cleaned_df.head().to_html(escape=False, index=False, classes='dataset-table'), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error during data cleaning: {str(e)}")
        else:
            st.info("Please upload your dataset in the sidebar first.")


    # ---------------------- RECOMMENDATION TAB ----------------------
    with tabs[1]:
        st.header("Model Recommendation")
        if 'df' in st.session_state and 'credentials_valid' in st.session_state:
            if st.button("Analyze Dataset"):
                with st.spinner("Analyzing dataset..."):
                    try:
                        temp_csv_path = "temp_dataset.csv"
                        st.session_state['df'].to_csv(temp_csv_path, index=False)
                        ralf = Ralf(HF_TOKEN=hf_token,
                                    OPENAI_API_KEY=openai_key,
                                    GEMINI_API_KEY=gemini_key)
                        llm_df, dataset_df, analysis = ralf.recommend(
                            temp_csv_path,
                            st.session_state['source_col'],
                            st.session_state['target_col']
                        )
                        os.remove(temp_csv_path)
                        st.session_state['llm_df'] = llm_df
                        st.session_state['dataset_df'] = dataset_df
                        st.session_state['analysis'] = analysis
                        st.success("Analysis complete! Check model recommendations below.")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

        if 'analysis' in st.session_state:
            st.subheader("🧠 Problem Analysis")
            raw_analysis = st.session_state['analysis']
            if isinstance(raw_analysis, str):
                try:
                    analysis = json.loads(raw_analysis)
                except json.JSONDecodeError:
                    analysis = {"reasoning": raw_analysis}
            else:
                analysis = raw_analysis
            st.markdown(f"**📝 Task Type:** `{analysis.get('types', ['Not specified'])[0]}`")
            st.info(analysis.get("reasoning", "No reasoning provided"))

        if 'llm_df' in st.session_state and not st.session_state['llm_df'].empty:
            st.subheader("Recommended Models")
            llm_df = st.session_state['llm_df'].copy()
            if "Name" in llm_df.columns and "Hugging Face URL" in llm_df.columns:
                llm_df["Name"] = llm_df.apply(
                    lambda row: f'<a href="{row["Hugging Face URL"]}" target="_blank">{row["Name"]}</a>'
                    if pd.notnull(row["Hugging Face URL"]) and pd.notnull(row["Name"]) else row["Name"], axis=1
                )
            llm_df.insert(0, "S.No", range(1, len(llm_df) + 1))
            st.markdown(llm_df.to_html(escape=False, index=False, classes='dataset-table'), unsafe_allow_html=True)

        if 'dataset_df' in st.session_state and not st.session_state['dataset_df'].empty:
            st.subheader("Recommended Golden Datasets")
            dataset_df = st.session_state['dataset_df'].copy()
            if "Name" in dataset_df.columns and "URL" in dataset_df.columns:
                dataset_df["Name"] = dataset_df.apply(
                    lambda row: f'<a href="{row["URL"]}" target="_blank">{row["Name"]}</a>'
                    if pd.notnull(row["URL"]) and pd.notnull(row["Name"]) else row["Name"], axis=1
                )
            dataset_df.insert(0, "S.No", range(1, len(dataset_df) + 1))
            
            st.markdown(dataset_df.to_html(escape=False, index=False, classes='dataset-table'), unsafe_allow_html=True)


    # ---------------------- AUGMENTATION TAB ----------------------
    with tabs[2]:
        st.header("Model Training & Scoring")

        if 'df' in st.session_state and 'llm_df' in st.session_state:
            df = st.session_state['df']
            llm_df = st.session_state['llm_df']

            st.subheader("Original Dataset")
            st.markdown(df.head().to_html(escape=False, index=False, classes='dataset-table'), unsafe_allow_html=True)

            if st.button("Train & Score All Models"):
                from sklearn.model_selection import train_test_split
                results = []

            # Split into train & validation sets
                train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

                try:
                    with st.spinner("Training and scoring models..."):
                        for _, model_row in llm_df.iterrows():
                            model_name = model_row["Name"]
                            model_id = model_row.get("Model ID") or model_row.get("Hugging Face URL", "")

                            st.write(f"🔄 Training model: **{model_name}**")

                            try:
                                ralf = Ralf(
                                    HF_TOKEN=hf_token,
                                    OPENAI_API_KEY=openai_key,
                                    GEMINI_API_KEY=gemini_key
                                )

                            # Load and process data
                                ralf.load_and_process_data(
                                    train_df,
                                    st.session_state['source_col'],
                                    st.session_state['target_col'],
                                    model_id
                                )

                            # Initialize trainer with LoRA/fallback logic
                                ralf.initialize_trainer(model_id)

                            # Train the model
                                ralf.train()

                            # Evaluate on validation set
                                metrics = ralf.trainer.evaluate()
                                score = metrics.get("eval_accuracy") or metrics.get("eval_f1") or 0

                                results.append({
                                    "Model": model_name,
                                    "loss": metrics["eval_loss"],
                                    "accuracy": metrics["eval_accuracy"],
                                    "precision": metrics["eval_precision"],
                                    "recall": metrics["eval_recall"],
                                    "f1": metrics["eval_f1"],
                                    "Score": score
                                })

                            except Exception as e:
                                st.warning(f"⚠️ Skipping {model_name} due to error: {str(e)}")
                                continue

                    if results:
                    # Display sorted results
                        results_df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
                        st.subheader("Model Scores")
                        st.dataframe(results_df)

                    # Highlight best model
                        best_model = results_df.iloc[0]
                        st.success(f"🏆 Most reliable model: {best_model['Model']} with score {best_model['Score']}")
                    else:
                        st.error("No models were successfully trained.")

                except Exception as e:
                    st.error(f"Unexpected error during training/scoring: {str(e)}")

        else:
            st.info("Please upload a dataset and get model recommendations first.")


    # ---------------------- LUSTRATION TAB ----------------------
    with tabs[3]:
        st.header("Data Lustration(Cleaning & Preprocessing)")
        #if 'df' in st.session_state:
            #df = st.session_state['df']
            #st.write("Original Dataset:")
            #st.markdown(df.head(10).to_html(escape=False, index=False, classes='dataset-table'), unsafe_allow_html=True)
            #remove_nulls = st.checkbox("Remove rows with missing values")
            #lowercase_text = st.checkbox("Convert text to lowercase")
            #remove_duplicates = st.checkbox("Remove duplicate rows")
            #if st.button("Clean Data"):
                #try:
                    #cleaned_df = df.copy()
                    #if remove_nulls:
                        #cleaned_df.dropna(inplace=True)
                    #if lowercase_text:
                        #cleaned_df[st.session_state['source_col']] = cleaned_df[st.session_state['source_col']].str.lower()
                        #cleaned_df[st.session_state['target_col']] = cleaned_df[st.session_state['target_col']].str.lower()
                    #if remove_duplicates:
                        #cleaned_df.drop_duplicates(inplace=True)
                    #st.session_state['cleaned_df'] = cleaned_df
                    #st.success("Data cleaned successfully!")
                    #st.markdown(cleaned_df.head(10).to_html(escape=False, index=False, classes='dataset-table'), unsafe_allow_html=True)
                #except Exception as e:
                    #st.error(f"Error during data cleaning: {str(e)}")
            #else:
                #st.info("Please upload and analyze your dataset first to apply lustration.")


    # ---------------------- TRAINING TAB ----------------------
    with tabs[4]:
        st.header("Model Futureproofing")
        if 'selected_models' in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.number_input("Number of epochs", min_value=1, value=3)
                batch_size = st.number_input("Batch size", min_value=1, value=16)
                st.session_state['epochs'] = epochs
                st.session_state['batch_size'] = batch_size
            with col2:
                learning_rate = st.number_input("Learning rate", min_value=0.0, value=2e-5)
                output_dir = st.text_input("Output directory", value="./results")
                st.session_state['learning_rate'] = learning_rate
                st.session_state['output_dir'] = output_dir

            if st.button("Start Training"):
                try:
                    with st.spinner("Training in progress..."):
                        progress_bar = st.progress(0)
                        ralf = Ralf(HF_TOKEN=hf_token, 
                                  OPENAI_API_KEY=openai_key, 
                                  GEMINI_API_KEY=gemini_key)
                        total_models = len(st.session_state['selected_models'])
                        for idx, model_id in enumerate(st.session_state['selected_models']):
                            ralf.load_and_process_data(
                                st.session_state['df'],
                                st.session_state['source_col'],
                                st.session_state['target_col'],
                                model_id
                            )
                            ralf.load_and_configure_model()
                            total_steps = epochs * (len(st.session_state['df']) // batch_size)
                            current_step = 0
                            def update_progress(step_info):
                                nonlocal current_step
                                current_step += 1
                                progress = min((idx + current_step / total_steps) / total_models, 1.0)
                                progress_bar.progress(progress)
                            ralf.trainer.add_callback(update_progress)
                            ralf.train()
                        progress_bar.progress(1.0)
                        st.success("Training completed for all recommended models!")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        else:
            st.info("Please analyze your dataset and get model recommendations first.")


if __name__ == "__main__":
    main()