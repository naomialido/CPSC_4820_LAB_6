import streamlit as st
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # Correct import for matplotlib
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb  # Import XGBoost
from sklearn import metrics
from PIL import Image
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file in the same directory as this script
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


#### Starting app
# title and details
st.title("Universal Classifier Application")
st.subheader(
    " This is a simple streamlit application where we can analyse ,transform and train different dataset with ease . Here we will be using different classifier models and evaluate ."
)
st.text(" ")
st.write(" #### lets explore the dataset !")

CHATBOT_SYSTEM_PROMPT = """
You are "ML Guide", a friendly assistant embedded inside a machine-learning
web application called the Universal Classifier App.

Your job is to help NON-TECHNICAL (layman) users:
1. Understand what each section of the app does (EDA, Visualization, Model Creation).
2. Understand every hyperparameter they can tune, in plain English, with simple
   analogies and practical advice on what values to try.
3. Interpret the data-analysis and model evaluation results shown in the app
   (accuracy, confusion matrix, precision, recall, F1, correlation heatmap, etc.)
   in a way that a total beginner can understand.

The models available are:
  KNN - hyperparameter: K (number of neighbours)
  Logistic Regression - hyperparameter: solver
  XGBoost - hyperparameters: n_estimators, max_depth, verbosity, booster, learning_rate
  CatBoost - hyperparameters: loss_function, eval_metric, random_seed

IMPORTANT: If the user asks about anything not related to this app, machine learning,
data analysis, or the models and hyperparameters listed above, you must respond with
exactly this message and nothing else:
"Sorry, I'm not sure if I understand. I can only help with questions about this app,
the machine learning models, hyperparameters, and how to read the results."

Always use simple everyday language, give concrete examples or analogies where
possible, be encouraging and patient, and keep replies concise.
"""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chatbot_open" not in st.session_state:
    st.session_state.chatbot_open = False
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None


def ask_groq(user_message, history):
    if not GROQ_API_KEY:
        return "No API key found. Add GROQ_API_KEY to your .env file."

    last_reply = history[-1]["content"] if history else ""
    cache_key = f"{user_message}||{last_reply}"
    if cache_key in st.session_state.response_cache:
        return st.session_state.response_cache[cache_key]

    try:
        client = Groq(api_key=GROQ_API_KEY)
        trimmed = history[-6:] if len(history) > 6 else history
        messages = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}]
        messages += trimmed
        messages.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=400,
            temperature=0.5,
        )
        reply = response.choices[0].message.content
        st.session_state.response_cache[cache_key] = reply
        return reply
    except Exception as e:
        err = str(e)
        if "invalid_api_key" in err.lower() or "authentication" in err.lower():
            return "API key invalid. Check GROQ_API_KEY in your .env file."
        if "rate_limit" in err.lower() or "429" in err:
            return "Rate limit reached, please wait a few seconds and try again."
        return f"Error: {err}"


with st.sidebar:
    toggle_label = (
        "ML Guide Chat  ▲" if st.session_state.chatbot_open else "ML Guide Chat  ▼"
    )
    if st.button(toggle_label, use_container_width=True):
        st.session_state.chatbot_open = not st.session_state.chatbot_open

    if st.session_state.chatbot_open:
        st.markdown("### ML Guide")
        st.caption(
            "Ask me anything about the app — hyperparameters, results, charts, and more."
        )

        if not GROQ_API_KEY:
            st.error("No API key found. Add GROQ_API_KEY to your .env file.")

        if st.session_state.pending_message:
            msg = st.session_state.pending_message
            st.session_state.pending_message = None
            reply = ask_groq(msg, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "user", "content": msg})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply}
            )

        st.markdown("**Quick questions:**")
        quick_questions = [
            "What does accuracy mean?",
            "How do I tune K in KNN?",
            "How do I read a confusion matrix?",
            "What is the learning rate in XGBoost?",
            "What does correlation tell me?",
        ]
        for qq in quick_questions:
            if st.button(qq, key=f"qq_{qq}", use_container_width=True):
                st.session_state.pending_message = qq
                st.rerun()

        if st.session_state.chat_history:
            for msg in st.session_state.chat_history[-10:]:
                role = msg["role"]
                text = msg["content"]
                if role == "user":
                    st.markdown(f"**You:** {text}")
                else:
                    st.markdown(f"**ML Guide:** {text}")

        user_q = st.text_area(
            "Ask your question:",
            placeholder="e.g. What should I set n_estimators to?",
            height=80,
            key="chat_input",
        )
        col_send, col_clear = st.columns([2, 1])
        with col_send:
            if st.button("Send", use_container_width=True):
                if user_q.strip():
                    st.session_state.pending_message = user_q.strip()
                    st.rerun()
        with col_clear:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.response_cache = {}
                st.rerun()


# suppressing warning
# st.set_option('deprecation.showPyplotGlobalUse', True)  # Optional, not necessary
# st.set_option('deprecation.showPyplotGlobalUse', False)  # Optional, not necessary

# adding sidebar
st.text(" ")
dataset_type = st.sidebar.selectbox(
    "Dataset Type :", ("Preloaded sklearn dataset", "Custom dataset")
)


# defining dataset loading function


def get_dataset(name):
    if name == "Iris data":
        data = datasets.load_iris()
    elif name == "Wine data":
        data = datasets.load_wine()
    else:
        data = datasets.load_digits(n_class=10)

    target = data["target"]
    df = pd.DataFrame(
        data=np.c_[data["data"], data["target"]],
        columns=data["feature_names"] + ["target"],
    )

    return df, target


df = None  # Use None instead of empty DataFrame

# session-state to hold multiple uploaded tables and join history --
if "tables" not in st.session_state:
    st.session_state.tables = {}  # {filename: DataFrame}
if "join_history" not in st.session_state:
    st.session_state.join_history = []  # list of join description strings

if dataset_type == "Custom dataset":
    # accept multiple CSV files instead of just one --
    st.markdown("### Upload Your Dataset(s)")
    st.info(
        "You may upload 2 or more CSV files. Each file becomes a separate table that can be joined."
    )
    uploaded_files = st.file_uploader(
        "Choose the dataset(s)", type="csv", accept_multiple_files=True
    )

    if uploaded_files:
        # Load every uploaded file into session-state tables
        for uf in uploaded_files:
            if uf.name not in st.session_state.tables:
                st.session_state.tables[uf.name] = pd.read_csv(uf)

        st.success(f"File(s) uploaded! {len(st.session_state.tables)} table(s) loaded.")

        # Quick preview of each uploaded table
        with st.expander("Preview uploaded tables", expanded=True):
            for tname, tdf in st.session_state.tables.items():
                st.markdown(
                    f"**Table: `{tname}`** - {tdf.shape[0]} rows x {tdf.shape[1]} cols"
                )
                st.dataframe(tdf.head(5))
                st.markdown("---")

        # TABLE JOIN SECTION
        if len(st.session_state.tables) >= 2:
            st.markdown("---")
            st.markdown("### Join Tables")

            # ask if user would like to join the tables
            do_join = st.radio(
                "Would you like to join two tables?", ("No", "Yes"), horizontal=True
            )

            if do_join == "Yes":
                table_names = list(st.session_state.tables.keys())

                # ask which table is Table A and which is Table B
                st.markdown("#### Step 1 - Choose Table A and Table B")
                col1, col2 = st.columns(2)
                with col1:
                    table_a_name = st.selectbox("Table A", table_names, key="ta")
                with col2:
                    remaining = [t for t in table_names if t != table_a_name]
                    table_b_name = st.selectbox("Table B", remaining, key="tb")

                table_a = st.session_state.tables[table_a_name]
                table_b = st.session_state.tables[table_b_name]

                # ask what key on Table A and the key on Table B
                st.markdown(
                    "#### Step 2 - Select the key on Table A and the key on Table B"
                )
                col3, col4 = st.columns(2)
                with col3:
                    key_a = st.selectbox(
                        f"Key column on Table A (`{table_a_name}`)",
                        table_a.columns.tolist(),
                        key="ka",
                    )
                with col4:
                    key_b = st.selectbox(
                        f"Key column on Table B (`{table_b_name}`)",
                        table_b.columns.tolist(),
                        key="kb",
                    )

                # ask the user to give a new name to the new table
                st.markdown("#### Step 3 - Give a new name to the joined table")
                new_table_name = st.text_input(
                    "New table name:", value="joined_table", key="new_tname"
                )

                # ask the user how to join the table (LEFT JOIN, RIGHT JOIN, INNER, OUTER)
                st.markdown("#### Step 4 - Join Type")
                join_type = st.selectbox(
                    "How would you like to join?",
                    ["inner", "left", "right", "outer"],
                    key="join_type",
                )
                join_desc = {
                    "inner": "INNER JOIN - only rows with matching keys in both tables",
                    "left": "LEFT JOIN  - all rows from Table A, matched rows from Table B",
                    "right": "RIGHT JOIN - all rows from Table B, matched rows from Table A",
                    "outer": "OUTER JOIN - all rows from both tables (full outer)",
                }
                st.caption(join_desc[join_type])

                # ask the user how to handle missing values
                st.markdown("#### Step 5 - Handle Missing Values (post-join)")
                missing_method_join = st.selectbox(
                    "How to handle NaN values introduced by the join?",
                    [
                        "Leave as NaN",
                        "Fill numeric with Mean",
                        "Fill numeric with Median",
                        "Fill numeric with 0",
                        "Fill all with Unknown / 0",
                        "Drop rows with any NaN",
                    ],
                    key="miss_method_join",
                )

                # after all input, join the dataframe by python
                if st.button("Perform Join", key="do_join_btn"):
                    joined = pd.merge(
                        table_a,
                        table_b,
                        left_on=key_a,
                        right_on=key_b,
                        how=join_type,
                        suffixes=("_A", "_B"),
                    )

                    # apply chosen missing-value strategy
                    if missing_method_join == "Fill numeric with Mean":
                        num_cols_j = joined.select_dtypes(include=np.number).columns
                        joined[num_cols_j] = joined[num_cols_j].fillna(
                            joined[num_cols_j].mean()
                        )
                    elif missing_method_join == "Fill numeric with Median":
                        num_cols_j = joined.select_dtypes(include=np.number).columns
                        joined[num_cols_j] = joined[num_cols_j].fillna(
                            joined[num_cols_j].median()
                        )
                    elif missing_method_join == "Fill numeric with 0":
                        num_cols_j = joined.select_dtypes(include=np.number).columns
                        joined[num_cols_j] = joined[num_cols_j].fillna(0)
                    elif missing_method_join == "Fill all with Unknown / 0":
                        for col in joined.columns:
                            if joined[col].dtype == object:
                                joined[col] = joined[col].fillna("Unknown")
                            else:
                                joined[col] = joined[col].fillna(0)
                    elif missing_method_join == "Drop rows with any NaN":
                        joined = joined.dropna()
                    # else: Leave as NaN - do nothing

                    # save the new joined table under the user-given name
                    final_name = (
                        new_table_name.strip()
                        if new_table_name.strip()
                        else "joined_table"
                    )
                    st.session_state.tables[final_name] = joined
                    st.session_state.join_history.append(
                        f"`{table_a_name}` {join_type.upper()} JOIN `{table_b_name}` "
                        f"on {key_a}={key_b} -> **{final_name}**"
                    )

                    st.success(
                        f"Join complete! New table '{final_name}' has "
                        f"{joined.shape[0]} rows x {joined.shape[1]} cols."
                    )

                    # after joining, show the top 10 lines to the user
                    st.markdown("**Top 10 rows of the joined table:**")
                    st.dataframe(joined.head(10))

                    # ask users if they want to further join the table if there are more than 2 tables
                    if len(st.session_state.tables) > 2:
                        st.info(
                            "You have more than 2 tables available. "
                            "You can perform another join above by choosing different tables."
                        )

            # show join history
            if st.session_state.join_history:
                with st.expander("Join History"):
                    for h in st.session_state.join_history:
                        st.markdown(f"- {h}")

    if not st.session_state.tables:
        st.warning("Please upload at least one CSV file to continue.")
        st.stop()

else:
    dataset_name = st.selectbox(
        "Select the dataset", ("Iris data", "Wine data", "digits data (upto 10)")
    )
    df, target = get_dataset(dataset_name)
    st.session_state.tables = {dataset_name: df}  # store in tables for consistency
    st.dataframe(df)


# in the main program, if there is only one table leave it as original.
# If there are more than one table, add one more option for the user to ask them which table they would use. --
st.markdown("---")
all_table_names = list(st.session_state.tables.keys())

if len(all_table_names) == 1:
    # only one table - leave it as the original
    active_table_name = all_table_names[0]
else:
    # more than one table - ask which one to use for analysis
    st.markdown("### Select Active Table for Analysis")
    active_table_name = st.selectbox(
        "Which table would you like to use for EDA / Visualization / Model Creation?",
        all_table_names,
    )

df = st.session_state.tables[active_table_name]
st.success(
    f"Active table: **{active_table_name}** - {df.shape[0]} rows x {df.shape[1]} cols"
)

# quick overview card and CSV export to complete the data platform --
with st.expander("Active Table Overview"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    st.dataframe(df.head(10))
    num_cols_ov = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_ov = df.select_dtypes(exclude=np.number).columns.tolist()
    st.markdown(
        f"**Numeric columns ({len(num_cols_ov)}):** {', '.join(num_cols_ov) if num_cols_ov else '-'}"
    )
    st.markdown(
        f"**Categorical columns ({len(cat_cols_ov)}):** {', '.join(cat_cols_ov) if cat_cols_ov else '-'}"
    )

# CSV download button in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Export Active Table")
csv_export = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download as CSV",
    data=csv_export,
    file_name=f"{active_table_name.replace('.csv', '')}_export.csv",
    mime="text/csv",
)


# creating a main sidebar function to handle different operations


def main():
    activities = ["Explanatory Data Analysis", "Visualization", "Model Creation"]
    option = st.sidebar.selectbox("Choose your operation : ", activities)
    df1 = pd.DataFrame({})  # df initialization for all !

    # EDA
    if option == "Explanatory Data Analysis":
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.subheader("Explanatory Data Analysis (EDA) ")
        st.text(" ")
        st.text(" ")
        if st.checkbox("Data Shape:"):
            st.write(df.shape)
        if st.checkbox("Data Columns Name:"):
            st.write(df.columns)
        st.text(" ")
        st.text(" ")
        st.write("#### Do you want to filter attributes for analysis ")
        st.text("select No to consider entire data set")
        ans1 = st.selectbox("response", ["Yes", "No"])

        if ans1 == "Yes":
            selected_col = st.multiselect("filter Columns : ", df.columns)
            st.text(" ")

            df1 = df[selected_col]
            if not df1.empty:
                st.text("selected dataframe")
                st.dataframe(df1)
                if st.checkbox(
                    "Data summary (categorical attributes are ignored here )"
                ):
                    st.write(df1.describe())
                st.text(" ")
                st.text(" ")
                if st.checkbox(
                    "Data Information with null values ,data types and shape"
                ):
                    buffer = io.StringIO()
                    df1.info(buf=buffer)
                    s = buffer.getvalue()
                    st.text(" ")
                    st.text(s)
                st.text(" ")
                st.text(" ")
                if st.checkbox("Data Correlation"):
                    st.write(df1.select_dtypes(include=np.number).corr())
        else:
            df1 = df.copy()
            st.text(" ")
            st.text("selected dataframe")
            st.dataframe(df1)
            st.text(" ")
            st.text(" ")
            if st.checkbox("Data summary (categorical attributes are ignored here )"):
                st.write(df1.describe())
            st.text(" ")
            st.text(" ")
            if st.checkbox("Data Information with null values ,data types and shape"):
                buffer = io.StringIO()
                df1.info(buf=buffer)
                s = buffer.getvalue()
                st.text(" ")
                st.text(s)
            st.text(" ")
            st.text(" ")
            if st.checkbox("Data Correlation"):
                st.write(df1.select_dtypes(include=np.number).corr())

    # Visaulization
    elif option == "Visualization":
        df1 = df.copy()
        st.write("#### Do you want to filter attributes for Visualization ")
        st.text("select No to consider entire data set")
        ans1 = st.selectbox("response", ["Yes", "No"])

        if ans1 == "Yes":
            selected_col = st.multiselect("filter Columns : ", df.columns)
            st.text(" ")

            df1 = df[selected_col]
            if not df1.empty:
                st.text("selected dataframe")
                st.dataframe(df1)
                if st.checkbox(
                    "Correlation Heatmap (categorical attributes are ignored here )"
                ):
                    num_df1 = df1.select_dtypes(include=np.number)
                    if not num_df1.empty:
                        fig, ax = plt.subplots()
                        sns.heatmap(
                            num_df1.corr(),
                            cmap="viridis",
                            square=True,
                            annot=True,
                            vmax=1,
                            ax=ax,
                        )
                        st.pyplot(fig)
                if st.checkbox("Pair Plot"):
                    num_df1 = df1.select_dtypes(include=np.number)
                    if not num_df1.empty:
                        fig = sns.pairplot(num_df1, diag_kind="kde")
                        st.pyplot(fig)
                if st.checkbox("Box Plot"):
                    for col in list(df1.select_dtypes(include=np.number).columns):
                        fig, ax = plt.subplots()
                        sns.boxplot(x=df1[col], ax=ax)
                        st.pyplot(fig)
                if st.checkbox("Count Plot Pie Chart"):
                    pie_columns = st.selectbox(
                        "Select Columns : ", df1.columns.to_list()
                    )
                    fig, ax = plt.subplots()
                    df1[pie_columns].value_counts().plot.pie(autopct="1.1f%%", ax=ax)
                    st.pyplot(fig)

        else:
            df1 = df.copy()
            st.text(" ")
            st.text("selected dataframe")
            st.dataframe(df1)
            st.text(" ")
            st.text(" ")
            if st.checkbox(
                "Correlation Heatmap (categorical attributes are ignored here )"
            ):
                sns.heatmap(df.corr(), cmap="viridis", square=True, annot=True, vmax=1)
                st.pyplot()
            if st.checkbox("Pair Plot"):
                sns.pairplot(df, diag_kind="kde")
                st.pyplot()
            if st.checkbox("Box Plot"):
                for col in df1.select_dtypes(include=np.number).columns:
                    fig, ax = plt.subplots()  # Create a figure and axis
                    sns.boxplot(x=df1[col], ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)  # Pass the correct figure

            if st.checkbox("Count Plot Pie Chart"):
                pie_columns = st.selectbox("Select Columns : ", df.columns.to_list())
                fig, ax = plt.subplots()
                df[pie_columns].value_counts().plot.pie(autopct="1.1f%%", ax=ax)
                st.pyplot(fig)

    # creating ML models
    elif option == "Model Creation":
        df1 = df.copy()
        st.write("#### Do you want to filter attributes for Model Creation ")
        st.text("select No to consider entire data set")
        ans1 = st.selectbox("response", ["Yes", "No"])

        if ans1 == "Yes":
            st.text("Always select the target variable at the end !")
            selected_col = st.multiselect("filter Columns : ", df.columns)
            st.text(" ")
            df1 = df[selected_col]
            missing_method = st.selectbox(
                "select method for missing values of numeric columns (generic)",
                ["mean", "median", "mode"],
            )
            if missing_method == "mean":
                df1 = df1.fillna(df1.select_dtypes(include=np.number).mean())
            elif missing_method == "median":
                df1 = df1.fillna(df1.select_dtypes(include=np.number).median())
            elif missing_method == "mode":
                df1 = df1.fillna(df1.mode().iloc[0])
            buffer = io.StringIO()
            df1.info(buf=buffer)
            s = buffer.getvalue()
            st.text(" ")
            st.text(s)
            if not df1.empty:
                st.text("selected dataframe")
                st.dataframe(df1)
                st.text(" ")
                st.text(" ")
                target_var = st.selectbox("Select the target variable", df1.columns)

                # model selection
                classifier_name = st.selectbox(
                    "Select the classifier : ",
                    ["KNN", "Logistic Regression", "XGBoost", "Catboost"],
                )

                # creating parameters
                def create_params(classifier_name):
                    param = {}
                    if classifier_name == "KNN":
                        param["n_neighbors"] = st.slider("K", 1, 20)
                    if classifier_name == "Logistic Regression":
                        param["solver"] = st.selectbox(
                            "Solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
                        )
                    if classifier_name == "Catboost":
                        param["loss_function"] = st.selectbox(
                            "loss function",
                            ["Logloss", "MAE", "CrossEntropy", "MultiClass"],
                        )
                        param["eval_metric"] = st.selectbox(
                            "eval_metric",
                            [
                                "Logloss",
                                "MAE",
                                "CrossEntropy",
                                "MultiClass",
                                "AUC",
                                "Precison",
                                "Recall",
                                "F1",
                                "Accuracy",
                            ],
                        )
                    if classifier_name == "XGBoost":
                        param["n_estimators"] = st.slider("n_estimators ", 10, 1000)
                        param["max_depth"] = st.slider("max_depth  ", 2, 10)
                        param["verbosity"] = st.slider("verbosity  ", 0, 3)
                        param["booster"] = st.selectbox(
                            "booster   ", ["gbtree", "gblinear", "dart"]
                        )
                        param["learning_rate"] = st.slider(
                            "learning_rate  ", 0.01, 5.00
                        )
                    return param

                param = create_params(classifier_name)

                ans2 = st.selectbox("Do you want to encode columns", [" ", "yes", "No"])
                # encoding
                if ans2 == "yes":
                    columns = st.multiselect("Select columns to encode", df1.columns)
                    if classifier_name == "Catboost":
                        st.text([df1.columns.get_loc(col) for col in columns])
                        if columns:
                            param["cat_features"] = [
                                df1.columns.get_loc(col) for col in columns
                            ]
                        else:
                            param["cat_features"] = None
                        # splitting df into x and y
                        X = df1.drop(target_var, axis=1)
                        y = df1[target_var]

                        # random seed
                        random_seed = st.slider("Select the random state : ", 1, 50)
                        param["random_seed"] = random_seed

                        # splitting it into train and test
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.25, random_state=random_seed
                        )

                        # standardize data
                        """if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                            X_train = StandardScaler().fit_transform(X_train)
                            X_test = StandardScaler().fit_transform(X_test)"""

                        # model
                        if st.checkbox("Train model "):
                            model = CatBoostClassifier(**param).fit(
                                X_train,
                                y_train,
                                eval_set=(X_test, y_test),
                                use_best_model=True,
                            )
                            # feature importance
                            st.dataframe(model.get_feature_importance(prettified=True))
                            sns.barplot(
                                data=model.get_feature_importance(prettified=True),
                                x="Feature Id",
                                y="Importances",
                            )
                            plt.xticks(rotation=45)
                            plt.title("Feature Importance")
                            st.pyplot()

                            st.write(
                                "ACCURACY : ",
                                metrics.accuracy_score(
                                    y_test, model.predict(X_test), average=None
                                ),
                            )

                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                fig, ax = plt.subplots()
                                sns.heatmap(
                                    metrics.confusion_matrix(
                                        y_test, model.predict(X_test)
                                    ),
                                    cmap="viridis",
                                    square=True,
                                    annot=True,
                                    vmax=1,
                                    ax=ax,
                                )
                                st.pyplot(fig)
                            if st.checkbox("Precision Score "):
                                st.write(
                                    "Precision Score :",
                                    metrics.precision_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("Recall Score "):
                                st.write(
                                    "Recall Score :",
                                    metrics.recall_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("F1 Score "):
                                st.write(
                                    "F1 Score :",
                                    metrics.f1_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )

                    else:
                        st.text([df1.columns.get_loc(col) for col in columns])
                        cat_features = [df1.columns.get_loc(col) for col in columns]
                        columnTransformer = ColumnTransformer(
                            [
                                (
                                    "encoder",
                                    OneHotEncoder(sparse_output=False),
                                    cat_features,
                                )
                            ],
                            remainder="passthrough",
                        )

                        # splitting df into x and y
                        X = df1.drop(target_var, axis=1)
                        y = df1[target_var]

                        # random seed
                        random_seed = st.slider("Select the random state : ", 1, 50)
                        param["random_seed"] = random_seed

                        # splitting it into train and test
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.25, random_state=random_seed
                        )
                        X_train = columnTransformer.fit_transform(X_train)
                        X_test = columnTransformer.transform(X_test)

                        # standardize data
                        if st.checkbox(
                            "Do you want to standardize the data (RECOMMENDED)"
                        ):
                            X_train = StandardScaler().fit_transform(X_train)
                            X_test = StandardScaler().fit_transform(X_test)

                        # model
                        if st.checkbox("Train model "):
                            if classifier_name == "XGBoost":
                                if classifier_name == "XGBoost":
                                    model = xgb.XGBClassifier(
                                        learning_rate=param["learning_rate"],
                                        n_estimators=param["n_estimators"],
                                        max_depth=param["max_depth"],
                                        verbosity=param["verbosity"],
                                        booster=param["booster"],
                                    ).fit(X_train, y_train)

                                    # Feature Importance
                                    fig, ax = plt.subplots()
                                    xgb.plot_importance(model, ax=ax)
                                    plt.xticks(rotation=45)
                                    st.pyplot(fig)  # Pass the correct figure

                                    st.write(
                                        "ACCURACY: ",
                                        metrics.accuracy_score(
                                            y_test, model.predict(X_test)
                                        ),
                                    )

                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    fig, ax = plt.subplots()
                                    sns.heatmap(
                                        metrics.confusion_matrix(
                                            y_test, model.predict(X_test)
                                        ),
                                        cmap="viridis",
                                        square=True,
                                        annot=True,
                                        vmax=1,
                                        ax=ax,
                                    )
                                    st.pyplot(fig)
                                if st.checkbox("Precision Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                if st.checkbox("Recall Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                    st.write(
                                        "Recall Score :",
                                        metrics.recall_score(
                                            y_test, model.predict(X_test)
                                        ),
                                    )
                                if st.checkbox("F1 Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                    st.write(
                                        "F1 Score :",
                                        metrics.f1_score(y_test, model.predict(X_test)),
                                    )

                            elif classifier_name == "Logistic Regression":
                                model = LogisticRegression(solver=param["solver"]).fit(
                                    X_train, y_train
                                )
                                st.write(
                                    "ACCURACY : ",
                                    metrics.accuracy_score(
                                        y_test, model.predict(X_test)
                                    ),
                                )
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    fig, ax = plt.subplots()
                                    sns.heatmap(
                                        metrics.confusion_matrix(
                                            y_test, model.predict(X_test)
                                        ),
                                        cmap="viridis",
                                        square=True,
                                        annot=True,
                                        vmax=1,
                                        ax=ax,
                                    )
                                    st.pyplot(fig)
                                if st.checkbox("Precision Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test)
                                        ),
                                    )
                                if st.checkbox("Recall Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                    st.write(
                                        "Recall Score :",
                                        metrics.recall_score(
                                            y_test, model.predict(X_test)
                                        ),
                                    )
                                if st.checkbox("F1 Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                    st.write(
                                        "F1 Score :",
                                        metrics.f1_score(y_test, model.predict(X_test)),
                                    )

                            else:
                                model = KNeighborsClassifier(
                                    n_neighbors=param["n_neighbors"]
                                ).fit(X_train, y_train)
                                st.write(
                                    "ACCURACY : ",
                                    metrics.accuracy_score(
                                        y_test, model.predict(X_test)
                                    ),
                                )
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    fig, ax = plt.subplots()
                                    sns.heatmap(
                                        metrics.confusion_matrix(
                                            y_test, model.predict(X_test)
                                        ),
                                        cmap="viridis",
                                        square=True,
                                        annot=True,
                                        vmax=1,
                                        ax=ax,
                                    )
                                    st.pyplot(fig)
                                if st.checkbox("Precision Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test)
                                        ),
                                    )
                                if st.checkbox("Recall Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                    st.write(
                                        "Recall Score :",
                                        metrics.recall_score(
                                            y_test, model.predict(X_test)
                                        ),
                                    )
                                if st.checkbox("F1 Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                    st.write(
                                        "F1 Score :",
                                        metrics.f1_score(y_test, model.predict(X_test)),
                                    )

                else:
                    if classifier_name == "Catboost":
                        # splitting df into x and y
                        X = df1.drop(target_var, axis=1)
                        y = df1[target_var]

                        # random seed
                        random_seed = st.slider("Select the random state : ", 1, 50)
                        param["random_seed"] = random_seed

                        # splitting it into train and test
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.25, random_state=random_seed
                        )

                        # standardize data
                        """if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                            X_train = StandardScaler().fit_transform(X_train)
                            X_test = StandardScaler().fit_transform(X_test)"""

                        # model
                        if st.checkbox("Train model "):
                            model = CatBoostClassifier(**param).fit(
                                X_train,
                                y_train,
                                eval_set=(X_test, y_test),
                                use_best_model=True,
                            )
                            # feature importance
                            st.dataframe(model.get_feature_importance(prettified=True))
                            sns.barplot(
                                data=model.get_feature_importance(prettified=True),
                                x="Feature Id",
                                y="Importances",
                            )
                            plt.xticks(rotation=45)
                            plt.title("Feature Importance")
                            st.pyplot()

                            st.write(
                                "ACCURACY : ",
                                metrics.accuracy_score(y_test, model.predict(X_test)),
                            )

                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                fig, ax = plt.subplots()
                                sns.heatmap(
                                    metrics.confusion_matrix(
                                        y_test, model.predict(X_test)
                                    ),
                                    cmap="viridis",
                                    square=True,
                                    annot=True,
                                    vmax=1,
                                    ax=ax,
                                )
                                st.pyplot(fig)
                            if st.checkbox("Precision Score "):
                                st.write(
                                    "Precision Score :",
                                    metrics.precision_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("Recall Score "):
                                st.write(
                                    "Recall Score :",
                                    metrics.recall_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("F1 Score "):
                                st.write(
                                    "F1 Score :",
                                    metrics.f1_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )

                    else:
                        # splitting df into x and y
                        X = df1.drop(target_var, axis=1)
                        y = df1[target_var]

                        # random seed
                        random_seed = st.slider("Select the random state : ", 1, 50)
                        param["random_seed"] = random_seed

                        # splitting it into train and test
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.25, random_state=random_seed
                        )

                        # standardize data
                        if st.checkbox(
                            "Do you want to standardize the data (RECOMMENDED)"
                        ):
                            X_train = StandardScaler().fit_transform(X_train)
                            X_test = StandardScaler().fit_transform(X_test)

                        # model
                        if st.checkbox("Train model "):
                            if classifier_name == "XGBoost":
                                model = xgb.XGBClassifier(
                                    learning_rate=param["learning_rate"],
                                    n_estimators=param["n_estimators"],
                                    max_depth=param["max_depth"],
                                    verbosity=param["verbosity"],
                                    booster=param["booster"],
                                ).fit(X_train, y_train)
                                # feature importance
                                fig, ax = plt.subplots()
                                xgb.plot_importance(model, ax=ax)
                                plt.xticks(rotation=45)
                                plt.title("Feature Importance")
                                st.pyplot(fig)

                                st.write(
                                    "ACCURACY : ",
                                    metrics.accuracy_score(
                                        y_test, model.predict(X_test)
                                    ),
                                )
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    fig, ax = plt.subplots()
                                    sns.heatmap(
                                        metrics.confusion_matrix(
                                            y_test, model.predict(X_test)
                                        ),
                                        cmap="viridis",
                                        square=True,
                                        annot=True,
                                        vmax=1,
                                        ax=ax,
                                    )
                                    st.pyplot(fig)
                                if st.checkbox("Precision Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                if st.checkbox("Recall Score "):
                                    st.write(
                                        "Recall Score :",
                                        metrics.recall_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                if st.checkbox("F1 Score "):
                                    st.write(
                                        "F1 Score :",
                                        metrics.f1_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )

                            elif classifier_name == "Logistic Regression":
                                model = LogisticRegression(solver=param["solver"]).fit(
                                    X_train, y_train
                                )
                                st.write(
                                    "ACCURACY : ",
                                    metrics.accuracy_score(
                                        y_test, model.predict(X_test)
                                    ),
                                )
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    fig, ax = plt.subplots()
                                    sns.heatmap(
                                        metrics.confusion_matrix(
                                            y_test, model.predict(X_test)
                                        ),
                                        cmap="viridis",
                                        square=True,
                                        annot=True,
                                        vmax=1,
                                        ax=ax,
                                    )
                                    st.pyplot(fig)
                                if st.checkbox("Precision Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                if st.checkbox("Recall Score "):
                                    st.write(
                                        "Recall Score :",
                                        metrics.recall_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                if st.checkbox("F1 Score "):
                                    st.write(
                                        "F1 Score :",
                                        metrics.f1_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )

                            else:
                                model = KNeighborsClassifier(
                                    n_neighbors=param["n_neighbors"]
                                ).fit(X_train, y_train)
                                st.write(
                                    "ACCURACY : ",
                                    metrics.accuracy_score(
                                        y_test, model.predict(X_test)
                                    ),
                                )
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    fig, ax = plt.subplots()
                                    sns.heatmap(
                                        metrics.confusion_matrix(
                                            y_test, model.predict(X_test)
                                        ),
                                        cmap="viridis",
                                        square=True,
                                        annot=True,
                                        vmax=1,
                                        ax=ax,
                                    )
                                    st.pyplot(fig)
                                if st.checkbox("Precision Score "):
                                    st.write(
                                        "Precision Score :",
                                        metrics.precision_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                if st.checkbox("Recall Score "):
                                    st.write(
                                        "Recall Score :",
                                        metrics.recall_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )
                                if st.checkbox("F1 Score "):
                                    st.write(
                                        "F1 Score :",
                                        metrics.f1_score(
                                            y_test, model.predict(X_test), average=None
                                        ),
                                    )

        else:
            df1 = df.copy()
            st.text(" ")
            st.text("selected dataframe")
            st.dataframe(df1)
            st.text(" ")
            st.text(" ")
            missing_method = st.selectbox(
                "select method for missing values of numeric columns (generic)",
                ["mean", "median", "mode"],
            )
            if missing_method == "mean":
                df1 = df1.fillna(df1.select_dtypes(include=np.number).mean())
            elif missing_method == "median":
                df1 = df1.fillna(df1.select_dtypes(include=np.number).median())
            elif missing_method == "mode":
                df1 = df1.fillna(df1.mode().iloc[0])
            buffer = io.StringIO()
            df1.info(buf=buffer)
            s = buffer.getvalue()
            st.text(" ")
            st.text(s)
            target_var = st.selectbox("Select the target variable", df1.columns)

            # model selection
            classifier_name = st.selectbox(
                "Select the classifier : ",
                ["KNN", "Logistic Regression", "XGBoost", "Catboost"],
            )

            # creating parameters
            def create_params(classifier_name):
                param = {}
                if classifier_name == "KNN":
                    param["n_neighbors"] = st.slider("K", 1, 20)
                if classifier_name == "Logistic Regression":
                    param["solver"] = st.selectbox(
                        "Solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
                    )
                if classifier_name == "Catboost":
                    param["loss_function"] = st.selectbox(
                        "loss function",
                        ["Logloss", "MAE", "CrossEntropy", "MultiClass"],
                    )
                    param["eval_metric"] = st.selectbox(
                        "eval_metric",
                        [
                            "Logloss",
                            "MAE",
                            "CrossEntropy",
                            "MultiClass",
                            "AUC",
                            "Precison",
                            "Recall",
                            "F1",
                            "Accuracy",
                        ],
                    )
                if classifier_name == "XGBoost":
                    param["n_estimators"] = st.slider("n_estimators ", 10, 1000)
                    param["max_depth"] = st.slider("max_depth  ", 2, 10)
                    param["verbosity"] = st.slider("verbosity  ", 0, 3)
                    param["booster"] = st.selectbox(
                        "booster   ", ["gbtree", "gblinear", "dart"]
                    )
                    param["learning_rate"] = st.slider("learning_rate  ", 0.01, 5.00)
                return param

            param = create_params(classifier_name)

            ans2 = st.selectbox("Do you want to encode columns", [" ", "yes", "No"])
            # encoding
            if ans2 == "yes":
                columns = st.multiselect("Select columns to encode", df1.columns)
                if classifier_name == "Catboost":
                    st.text([df1.columns.get_loc(col) for col in columns])
                    if columns:
                        param["cat_features"] = [
                            df1.columns.get_loc(col) for col in columns
                        ]
                    else:
                        param["cat_features"] = None
                    # splitting df into x and y
                    X = df1.drop(target_var, axis=1)
                    y = df1[target_var]

                    # random seed
                    random_seed = st.slider("Select the random state : ", 1, 50)
                    param["random_seed"] = random_seed

                    # splitting it into train and test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.25, random_state=random_seed
                    )

                    # standardize data
                    """if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                        X_train = StandardScaler().fit_transform(X_train)
                        X_test = StandardScaler().fit_transform(X_test)"""

                    # model
                    if st.checkbox("Train model "):
                        model = CatBoostClassifier(**param).fit(
                            X_train,
                            y_train,
                            eval_set=(X_test, y_test),
                            use_best_model=True,
                        )
                        # feature importance
                        st.dataframe(model.get_feature_importance(prettified=True))
                        fig, ax = plt.subplots()
                        sns.barplot(
                            data=model.get_feature_importance(prettified=True),
                            x="Feature Id",
                            y="Importances",
                            ax=ax,
                        )
                        plt.xticks(rotation=45)
                        plt.title("Feature Importance")
                        st.pyplot(fig)

                        st.write(
                            "ACCURACY : ",
                            metrics.accuracy_score(y_test, model.predict(X_test)),
                        )

                        st.text(" ")
                        st.text(" ")
                        if st.checkbox("Confusion Matrix "):
                            fig, ax = plt.subplots()
                            sns.heatmap(
                                metrics.confusion_matrix(y_test, model.predict(X_test)),
                                cmap="viridis",
                                square=True,
                                annot=True,
                                vmax=1,
                                ax=ax,
                            )
                            st.pyplot(fig)
                        if st.checkbox("Precision Score "):
                            st.write(
                                "Precision Score :",
                                metrics.precision_score(
                                    y_test, model.predict(X_test), average=None
                                ),
                            )
                        if st.checkbox("Recall Score "):
                            st.write(
                                "Recall Score :",
                                metrics.recall_score(
                                    y_test, model.predict(X_test), average=None
                                ),
                            )
                        if st.checkbox("F1 Score "):
                            st.write(
                                "F1 Score :",
                                metrics.f1_score(
                                    y_test, model.predict(X_test), average=None
                                ),
                            )

                else:
                    st.text([df1.columns.get_loc(col) for col in columns])
                    cat_features = [df1.columns.get_loc(col) for col in columns]
                    columnTransformer = ColumnTransformer(
                        [("encoder", OneHotEncoder(sparse_output=False), cat_features)],
                        remainder="passthrough",
                    )

                    # splitting df into x and y
                    X = df1.drop(target_var, axis=1)
                    y = df1[target_var]

                    # random seed
                    random_seed = st.slider("Select the random state : ", 1, 50)
                    param["random_seed"] = random_seed

                    # splitting it into train and test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.25, random_state=random_seed
                    )
                    X_train = columnTransformer.fit_transform(X_train)
                    X_test = columnTransformer.transform(X_test)

                    # standardize data
                    if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                        scaler = StandardScaler(with_mean=False)
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    # model
                    if st.checkbox("Train model "):
                        if classifier_name == "XGBoost":
                            model = xgb.XGBClassifier(
                                learning_rate=param["learning_rate"],
                                n_estimators=param["n_estimators"],
                                max_depth=param["max_depth"],
                                verbosity=param["verbosity"],
                                booster=param["booster"],
                            ).fit(X_train, y_train)
                            # feature importance

                            fig, ax = plt.subplots()
                            xgb.plot_importance(model, ax=ax)
                            plt.xticks(rotation=45)
                            plt.title("Feature Importance")
                            st.pyplot(fig)

                            st.write(
                                "ACCURACY : ",
                                metrics.accuracy_score(y_test, model.predict(X_test)),
                            )
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                fig, ax = plt.subplots()
                                sns.heatmap(
                                    metrics.confusion_matrix(
                                        y_test, model.predict(X_test)
                                    ),
                                    cmap="viridis",
                                    square=True,
                                    annot=True,
                                    vmax=1,
                                    ax=ax,
                                )
                                st.pyplot(fig)
                            if st.checkbox("Precision Score "):
                                st.write(
                                    "Precision Score :",
                                    metrics.precision_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("Recall Score "):
                                st.write(
                                    "Recall Score :",
                                    metrics.recall_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("F1 Score "):
                                st.write(
                                    "F1 Score :",
                                    metrics.f1_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )

                        elif classifier_name == "Logistic Regression":
                            model = LogisticRegression(solver=param["solver"]).fit(
                                X_train, y_train
                            )

                            st.write(
                                "ACCURACY: ",
                                metrics.accuracy_score(y_test, model.predict(X_test)),
                            )

                            # Confusion Matrix
                            if st.checkbox("Confusion Matrix"):
                                fig, ax = plt.subplots()
                                sns.heatmap(
                                    metrics.confusion_matrix(
                                        y_test, model.predict(X_test)
                                    ),
                                    cmap="viridis",
                                    annot=True,
                                    ax=ax,
                                )
                                st.pyplot(fig)

                            if st.checkbox("Precision Score "):
                                st.write(
                                    "Precision Score :",
                                    metrics.precision_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("Recall Score "):
                                st.write(
                                    "Recall Score :",
                                    metrics.recall_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("F1 Score "):
                                st.write(
                                    "F1 Score :",
                                    metrics.f1_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )

                        else:
                            model = KNeighborsClassifier(
                                n_neighbors=param["n_neighbors"]
                            ).fit(X_train, y_train)
                            st.write(
                                "ACCURACY : ",
                                metrics.accuracy_score(y_test, model.predict(X_test)),
                            )
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                fig, ax = plt.subplots()
                                sns.heatmap(
                                    metrics.confusion_matrix(
                                        y_test, model.predict(X_test)
                                    ),
                                    cmap="viridis",
                                    square=True,
                                    annot=True,
                                    vmax=1,
                                    ax=ax,
                                )
                                st.pyplot(fig)
                            if st.checkbox("Precision Score "):
                                st.write(
                                    "Precision Score :",
                                    metrics.precision_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("Recall Score "):
                                st.write(
                                    "Recall Score :",
                                    metrics.recall_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("F1 Score "):
                                st.write(
                                    "F1 Score :",
                                    metrics.f1_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )

            else:
                if classifier_name == "Catboost":
                    # splitting df into x and y
                    X = df1.drop(target_var, axis=1)
                    y = df1[target_var]

                    # random seed
                    random_seed = st.slider("Select the random state : ", 1, 50)
                    param["random_seed"] = random_seed

                    # splitting it into train and test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.25, random_state=random_seed
                    )

                    # standardize data
                    """if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                        X_train = StandardScaler().fit_transform(X_train)
                        X_test = StandardScaler().fit_transform(X_test)"""

                    # model
                    if st.checkbox("Train model "):
                        model = CatBoostClassifier(**param).fit(
                            X_train,
                            y_train,
                            eval_set=(X_test, y_test),
                            use_best_model=True,
                        )
                        # feature importance
                        st.dataframe(model.get_feature_importance(prettified=True))
                        fig, ax = plt.subplots()
                        sns.barplot(
                            data=model.get_feature_importance(prettified=True),
                            x="Feature Id",
                            y="Importances",
                            ax=ax,
                        )
                        plt.xticks(rotation=45)
                        plt.title("Feature Importance")
                        st.pyplot(fig)

                        st.write(
                            "ACCURACY : ",
                            metrics.accuracy_score(y_test, model.predict(X_test)),
                        )
                        st.text(" ")
                        st.text(" ")
                        if st.checkbox("Confusion Matrix "):
                            fig, ax = plt.subplots()
                            sns.heatmap(
                                metrics.confusion_matrix(y_test, model.predict(X_test)),
                                cmap="viridis",
                                square=True,
                                annot=True,
                                vmax=1,
                                ax=ax,
                            )
                            st.pyplot(fig)
                        if st.checkbox("Precision Score "):
                            st.write(
                                "Precision Score :",
                                metrics.precision_score(
                                    y_test, model.predict(X_test), average=None
                                ),
                            )
                        if st.checkbox("Recall Score "):
                            st.write(
                                "Recall Score :",
                                metrics.recall_score(
                                    y_test, model.predict(X_test), average=None
                                ),
                            )
                        if st.checkbox("F1 Score "):
                            st.write(
                                "F1 Score :",
                                metrics.f1_score(
                                    y_test, model.predict(X_test), average=None
                                ),
                            )

                else:
                    # splitting df into x and y
                    X = df1.drop(target_var, axis=1)
                    y = df1[target_var]

                    # random seed
                    random_seed = st.slider("Select the random state : ", 1, 50)
                    param["random_seed"] = random_seed

                    # splitting it into train and test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.25, random_state=random_seed
                    )

                    # standardize data
                    if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                        scaler = StandardScaler(with_mean=False)
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    # model
                    if st.checkbox("Train model "):
                        if classifier_name == "XGBoost":
                            model = xgb.XGBClassifier(
                                learning_rate=param["learning_rate"],
                                n_estimators=param["n_estimators"],
                                max_depth=param["max_depth"],
                                verbosity=param["verbosity"],
                                booster=param["booster"],
                            ).fit(X_train, y_train)
                            # feature importance

                            fig, ax = plt.subplots()
                            xgb.plot_importance(model, ax=ax)
                            plt.xticks(rotation=45)
                            plt.title("Feature Importance")
                            st.pyplot(fig)

                            st.write(
                                "ACCURACY : ",
                                metrics.accuracy_score(y_test, model.predict(X_test)),
                            )
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                fig, ax = plt.subplots()
                                sns.heatmap(
                                    metrics.confusion_matrix(
                                        y_test, model.predict(X_test)
                                    ),
                                    cmap="viridis",
                                    square=True,
                                    annot=True,
                                    vmax=1,
                                    ax=ax,
                                )
                                st.pyplot(fig)
                            if st.checkbox("Precision Score "):
                                st.write(
                                    "Precision Score :",
                                    metrics.precision_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("Recall Score "):
                                st.write(
                                    "Recall Score :",
                                    metrics.recall_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("F1 Score "):
                                st.write(
                                    "F1 Score :",
                                    metrics.f1_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )

                        elif classifier_name == "Logistic Regression":
                            model = LogisticRegression(solver=param["solver"]).fit(
                                X_train, y_train
                            )
                            st.write(
                                "ACCURACY : ",
                                metrics.accuracy_score(y_test, model.predict(X_test)),
                            )
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                fig, ax = plt.subplots()
                                sns.heatmap(
                                    metrics.confusion_matrix(
                                        y_test, model.predict(X_test)
                                    ),
                                    cmap="viridis",
                                    square=True,
                                    annot=True,
                                    vmax=1,
                                    ax=ax,
                                )
                                st.pyplot(fig)
                            if st.checkbox("Precision Score "):
                                st.write(
                                    "Precision Score :",
                                    metrics.precision_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("Recall Score "):
                                st.write(
                                    "Recall Score :",
                                    metrics.recall_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("F1 Score "):
                                st.write(
                                    "F1 Score :",
                                    metrics.f1_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )

                        else:
                            model = KNeighborsClassifier(
                                n_neighbors=param["n_neighbors"]
                            ).fit(X_train, y_train)
                            st.write(
                                "ACCURACY : ",
                                metrics.accuracy_score(y_test, model.predict(X_test)),
                            )
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                fig, ax = plt.subplots()
                                sns.heatmap(
                                    metrics.confusion_matrix(
                                        y_test, model.predict(X_test)
                                    ),
                                    cmap="viridis",
                                    square=True,
                                    annot=True,
                                    vmax=1,
                                    ax=ax,
                                )
                                st.pyplot(fig)
                            if st.checkbox("Precision Score "):
                                st.write(
                                    "Precision Score :",
                                    metrics.precision_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("Recall Score "):
                                st.write(
                                    "Recall Score :",
                                    metrics.recall_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )
                            if st.checkbox("F1 Score "):
                                st.write(
                                    "F1 Score :",
                                    metrics.f1_score(
                                        y_test, model.predict(X_test), average=None
                                    ),
                                )


# main function called

if __name__ == "__main__":
    if df is not None:  # Ensure df is loaded
        main()
    else:
        st.warning("Please upload a dataset or select a preloaded one.")
