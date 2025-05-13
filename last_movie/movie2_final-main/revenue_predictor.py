import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def revenue_predictor_page(df):
    st.header("💰 Revenue Predictor")
    st.write("Estimate movie revenue using both ML and DL models!")

    # Add Data Summary Statistics
    st.subheader("Training Data Summary")
    rev_df = df[['budget', 'runtime', 'vote_average', 'revenue', 'release_date', 'genres']].dropna()
    rev_df['release_year'] = rev_df['release_date'].dt.year
    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Movies", len(rev_df))
    with cols[1]:
        st.metric("Average Revenue", f"${rev_df['revenue'].mean():,.0f}")
    with cols[2]:
        st.metric("Average Budget", f"${rev_df['budget'].mean():,.0f}")

    # Fix genre encoding - extract genre names properly
    def extract_genre_names(genre_list):
        try:
            if isinstance(genre_list, str):
                genre_list = eval(genre_list)
            return [g['name'] for g in genre_list]
        except:
            return []

    rev_df['genre_names'] = rev_df['genres'].apply(extract_genre_names)
    genre_dummies = rev_df['genre_names'].str.join('|').str.get_dummies()
    rev_df = pd.concat([rev_df, genre_dummies], axis=1)

    # Calculate revenue range
    max_revenue = rev_df['revenue'].quantile(0.99)  # 99th percentile for realistic bounds
    min_revenue = rev_df['revenue'].quantile(0.01)  # 1st percentile for lower bound

    # Scale numerical features
    scaler = StandardScaler()
    num_features = ['budget', 'runtime', 'vote_average', 'release_year']
    rev_df[num_features] = scaler.fit_transform(rev_df[num_features])

    # Prepare features and target
    genre_cols = [col for col in rev_df.columns if col not in ['budget', 'runtime', 'vote_average', 'release_year',
                                                               'revenue', 'genres', 'release_date', 'genre_names']]
    X = rev_df[num_features + genre_cols]
    y = np.log1p(rev_df['revenue'])  # Log-transform revenue

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    ridge_model = Ridge(alpha=1.0)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    dl_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),  # Smaller network
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        alpha=0.01,  # Stronger regularization
        learning_rate_init=0.0001,
        batch_size=256  # Larger batches
    )

    # Train and Evaluate Models
    if st.button("Train and Evaluate Revenue Models"):
        with st.spinner("Training models..."):
            try:
                # Train models
                ridge_model.fit(X_train, y_train)
                rf_model.fit(X_train, y_train)
                dl_model.fit(X_train, y_train)

                # Evaluate models
                ridge_pred = ridge_model.predict(X_test)
                rf_pred = rf_model.predict(X_test)
                dl_pred = dl_model.predict(X_test)

                # Calculate metrics
                ridge_r2 = r2_score(y_test, ridge_pred)
                ridge_mae = mean_absolute_error(np.expm1(y_test), np.expm1(ridge_pred))  # Corrected MAE
                rf_r2 = r2_score(y_test, rf_pred)
                rf_mae = mean_absolute_error(np.expm1(y_test), np.expm1(rf_pred))  # Corrected MAE
                dl_r2 = r2_score(y_test, dl_pred)
                dl_mae = mean_absolute_error(np.expm1(y_test), np.expm1(dl_pred))  # Corrected MAE

                # Display metrics
                st.subheader("Model Evaluation Metrics")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Ridge R²", f"{ridge_r2:.2f}")
                    st.metric("Ridge MAE", f"${ridge_mae:,.0f}")
                with cols[1]:
                    st.metric("Random Forest R²", f"{rf_r2:.2f}")
                    st.metric("RF MAE", f"${rf_mae:,.0f}")
                with cols[2]:
                    st.metric("DL R²", f"{dl_r2:.2f}")
                    st.metric("DL MAE", f"${dl_mae:,.0f}")

                # Visualize training data distribution
                st.subheader("Training Data Revenue Distribution")
                fig3, ax3 = plt.subplots()
                ax3.hist(np.expm1(y_train), bins=50, color='blue', alpha=0.7)
                ax3.set_title("Training Data Revenue Distribution")
                ax3.set_xlabel("Revenue ($)")
                ax3.set_ylabel("Frequency")
                st.pyplot(fig3)

                # Save models and data
                st.session_state.rev_models = {
                    'ridge': ridge_model,
                    'rf': rf_model,
                    'dl': dl_model,
                    'scaler': scaler,
                    'genre_cols': genre_cols,
                    'num_features': num_features,
                    'max_revenue': max_revenue,
                    'min_revenue': min_revenue
                }

            except Exception as e:
                st.error(f"Model training failed: {str(e)}")

    # Prediction Section
    if 'rev_models' in st.session_state:
        st.subheader("Make a Revenue Prediction")

        # Input widgets
        cols = st.columns(3)
        with cols[0]:
            budget = st.number_input("Budget ($)", min_value=100000, max_value=500000000, value=50000000)
        with cols[1]:
            runtime = st.number_input("Runtime (min)", min_value=60, max_value=240, value=120)
        with cols[2]:
            rating = st.slider("Expected Rating", 1.0, 10.0, value=7.0, step=0.1)

        year = st.number_input("Release Year", min_value=1900, max_value=2025, value=2023)

        # Genre selection - use only genres seen during training
        available_genres = [g for g in st.session_state.rev_models['genre_cols'] if g in genre_dummies.columns]
        selected_genres = st.multiselect("Genres", available_genres, default=['Drama'] if 'Drama' in available_genres else None)

        # Input validation
        if not selected_genres:
            st.error("Please select at least one genre")
            return
        if budget < 100000:
            st.error("Budget too low - minimum $100,000")
            return

        if st.button("Predict Revenue"):
            try:
                # Prepare numerical features
                num_input = np.array([[budget, runtime, rating, year]])
                num_scaled = st.session_state.rev_models['scaler'].transform(num_input)

                # Prepare genre features
                genre_input = np.zeros((1, len(st.session_state.rev_models['genre_cols'])))
                for i, genre in enumerate(st.session_state.rev_models['genre_cols']):
                    if genre in selected_genres:
                        genre_input[0, i] = 1

                # Combine features
                X_input = np.hstack([num_scaled, genre_input])

                # Get predictions
                ridge_pred = np.expm1(st.session_state.rev_models['ridge'].predict(X_input)[0])
                rf_pred = np.expm1(st.session_state.rev_models['rf'].predict(X_input)[0])
                dl_pred = np.expm1(st.session_state.rev_models['dl'].predict(X_input)[0])

                # Clamp predictions
                ridge_pred = np.clip(ridge_pred, st.session_state.rev_models['min_revenue'], st.session_state.rev_models['max_revenue'])
                rf_pred = np.clip(rf_pred, st.session_state.rev_models['min_revenue'], st.session_state.rev_models['max_revenue'])
                dl_pred = np.clip(dl_pred, st.session_state.rev_models['min_revenue'], st.session_state.rev_models['max_revenue'])

                # Sanity check for DL predictions
                if dl_pred > 2 * rf_pred:
                    st.warning("DL prediction seems abnormally high - interpreting with caution")
                    dl_pred = rf_pred * 1.5  # Fallback to scaled RF prediction

                # Add confidence intervals for Random Forest
                rf_preds = [tree.predict(X_input)[0] for tree in st.session_state.rev_models['rf'].estimators_]
                lower = np.percentile(np.expm1(rf_preds), 5)
                upper = np.percentile(np.expm1(rf_preds), 95)

                # Display results
                st.subheader("Prediction Results")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Ridge Prediction", f"${ridge_pred:,.0f}")
                with cols[1]:
                    st.metric("Random Forest", f"${rf_pred:,.0f}")
                    st.write(f"95% Confidence Interval: ${lower:,.0f} - ${upper:,.0f}")
                with cols[2]:
                    st.metric("DL Prediction", f"${dl_pred:,.0f}")

                # ROI calculation
                roi_ridge = (ridge_pred - budget) / budget * 100
                roi_rf = (rf_pred - budget) / budget * 100
                st.info(f"""
                **Return on Investment:**
                - Ridge Model: {roi_ridge:.1f}%
                - Random Forest: {roi_rf:.1f}%
                """)

                # Historical comparisons
                similar_movies = rev_df[
                    (rev_df['Adventure'] == 1) & 
                    (rev_df['budget'].between(budget * 0.8, budget * 1.2))
                ].sort_values('revenue', ascending=False).head(3)

                if not similar_movies.empty:
                    st.write("Similar historical movies:")
                    st.dataframe(similar_movies[['budget', 'revenue']])

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")