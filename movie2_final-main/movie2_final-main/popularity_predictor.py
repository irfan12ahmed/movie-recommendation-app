
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def popularity_predictor_page(df):
    st.header("📈 Popularity Predictor")
    st.write("Estimate how popular a movie might be using both ML and DL models!")

    # Preprocess for Popularity Model
    pop_df = df[['budget', 'runtime', 'vote_average', 'popularity', 'release_date', 'genres']].dropna()
    pop_df['release_year'] = pop_df['release_date'].dt.year
    pop_df = pop_df.join(pop_df['genres'].str.get_dummies(sep=', '))

    # Calculate max popularity from training data for clamping
    try:
        max_popularity = np.nanmax(np.expm1(pop_df['popularity'])) * 1.5  # 50% buffer
        min_popularity = np.nanmin(np.expm1(pop_df['popularity'])) * 0.5  # 50% buffer
        
        # Validate the calculated limits
        if not np.isfinite(max_popularity) or max_popularity <= 0:
            max_popularity = 1000  # Default fallback value
        if not np.isfinite(min_popularity) or min_popularity < 0:
            min_popularity = 0
        
        # Store in session state
        st.session_state.max_popularity = max_popularity
        st.session_state.min_popularity = min_popularity

    except Exception as e:
        st.error(f"Error calculating popularity range: {str(e)}")
        # Set safe defaults
        st.session_state.max_popularity = 1000
        st.session_state.min_popularity = 0

    # Scale numerical features
    scaler = StandardScaler()
    pop_df[['budget', 'runtime', 'vote_average', 'release_year']] = scaler.fit_transform(
        pop_df[['budget', 'runtime', 'vote_average', 'release_year']]
    )

    # Log-transform the target variable (popularity)
    y_pop = np.log1p(pop_df['popularity'])
    X_pop = pop_df.drop(columns=['popularity', 'genres', 'release_date'])
    X_train_pop, X_test_pop, y_train_pop, y_test_pop = train_test_split(X_pop, y_pop, test_size=0.2, random_state=42)

    # Initialize Models with improved parameters
    ridge_model = Ridge(alpha=1.0)
    lasso_model = Lasso(alpha=0.1)
    dl_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=2000,
        learning_rate_init=0.001,
        activation='relu',
        early_stopping=True,
        random_state=42
    )

    # Train and Evaluate Models
    if st.button("Train and Evaluate Models"):
        with st.spinner("Training models..."):
            # Train Ridge and Lasso Models
            ridge_model.fit(X_train_pop, y_train_pop)
            lasso_model.fit(X_train_pop, y_train_pop)

            # Train DL Model with progress updates
            dl_model.fit(X_train_pop, y_train_pop)

            # Evaluate Models
            ridge_preds = ridge_model.predict(X_test_pop)
            lasso_preds = lasso_model.predict(X_test_pop)
            dl_preds = dl_model.predict(X_test_pop)

            # Calculate metrics
            ridge_r2 = r2_score(y_test_pop, ridge_preds)
            ridge_mae = mean_absolute_error(y_test_pop, ridge_preds)
            lasso_r2 = r2_score(y_test_pop, lasso_preds)
            lasso_mae = mean_absolute_error(y_test_pop, lasso_preds)
            dl_r2 = r2_score(y_test_pop, dl_preds)
            dl_mae = mean_absolute_error(y_test_pop, dl_preds)

            # Display Model Evaluation Metrics
            st.subheader("Model Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Ridge R² Score", f"{ridge_r2:.2f}")
                st.metric("Ridge MAE", f"{ridge_mae:.2f}")
            
            with col2:
                st.metric("Lasso R² Score", f"{lasso_r2:.2f}")
                st.metric("Lasso MAE", f"{lasso_mae:.2f}")
            
            with col3:
                st.metric("DL R² Score", f"{dl_r2:.2f}")
                st.metric("DL MAE", f"{dl_mae:.2f}")

            # Save trained models and scaler
            st.session_state.models_trained = True
            st.session_state.ridge_model = ridge_model
            st.session_state.lasso_model = lasso_model
            st.session_state.dl_model = dl_model
            st.session_state.scaler = scaler
            st.session_state.genre_cols = [col for col in X_pop.columns if col not in ['budget', 'runtime', 'vote_average', 'release_year']]
            st.session_state.max_year = pop_df['release_year'].max()

            # Visualizations
            st.subheader("Model Diagnostics")
            
            # Actual vs Predicted Plot
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            try:
                # Ensure we have valid data for plotting
                valid_mask = ~np.isinf(np.expm1(y_test_pop)) & ~np.isnan(np.expm1(y_test_pop))
                y_test_valid = np.expm1(y_test_pop[valid_mask])
                ridge_preds_valid = np.expm1(ridge_preds[valid_mask])
                dl_preds_valid = np.expm1(dl_preds[valid_mask])
                
                ax1.scatter(y_test_valid, ridge_preds_valid, label="Ridge", alpha=0.6)
                ax1.scatter(y_test_valid, dl_preds_valid, label="DL", alpha=0.6)
                
                # Use calculated max or fallback to 1000
                plot_max = min(st.session_state.max_popularity, 1000) if hasattr(st.session_state, 'max_popularity') else 1000
                ax1.plot([0, plot_max], [0, plot_max], 'k--', label="Perfect Prediction")
                ax1.set_title("Predictions vs Actual")
                ax1.set_xlabel("Actual Popularity")
                ax1.set_ylabel("Predicted Popularity")
                ax1.legend()
                ax1.set_xlim(0, plot_max)
                ax1.set_ylim(0, plot_max)
                st.pyplot(fig1)
            except Exception as e:
                st.error(f"Could not generate prediction plot: {str(e)}")

            # Feature Importance
            coef_df = pd.DataFrame({
                'feature': X_train_pop.columns,
                'ridge_coef': ridge_model.coef_,
                'lasso_coef': lasso_model.coef_
            }).sort_values('ridge_coef', ascending=False)
            
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            coef_df.plot(x='feature', y=['ridge_coef', 'lasso_coef'], kind='bar', ax=ax2)
            ax2.set_title("Feature Importance")
            ax2.set_ylabel("Coefficient Value")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig2)

    # Prediction Section
    if st.session_state.get('models_trained', False):
        st.subheader("Make a Prediction")
        
        # Input widgets with better defaults
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_budget = st.number_input(
                "Budget (USD)", 
                min_value=100000, 
                max_value=500000000,
                value=50000000,
                step=1000000,
                help="Typical range: $1M-$200M"
            )
        
        with col2:
            user_runtime = st.number_input(
                "Runtime (minutes)", 
                min_value=60, 
                max_value=240,
                value=120,
                help="Typical range: 80-180 minutes"
            )
        
        with col3:
            user_vote_avg = st.slider(
                "Expected Rating", 
                1.0, 10.0, 
                value=7.0,
                step=0.1,
                help="Typical range: 4.0-9.0"
            )
        
        user_genre = st.selectbox(
            "Primary Genre", 
            st.session_state.genre_cols,
            index=st.session_state.genre_cols.index('Animation') if 'Animation' in st.session_state.genre_cols else 0
        )
        
        year_input = st.number_input(
            "Release Year",
            min_value=1900,
            max_value=2025,
            value=2023
        )

        # Prediction button
        if st.button("Predict Popularity"):
            try:
                # Prepare input data
                user_numeric = np.array([[user_budget, user_runtime, user_vote_avg, year_input]])
                user_numeric_scaled = st.session_state.scaler.transform(user_numeric)
                user_genre_array = np.array([[1 if user_genre == g else 0 for g in st.session_state.genre_cols]])
                input_pop = np.hstack([user_numeric_scaled, user_genre_array])

                # Get predictions with confidence estimates
                ridge_pred_log = st.session_state.ridge_model.predict(input_pop)[0]
                dl_pred_log = st.session_state.dl_model.predict(input_pop)[0]

                # Transform and clamp predictions
                ridge_pred = np.clip(np.expm1(ridge_pred_log), 
                                   st.session_state.min_popularity, 
                                   st.session_state.max_popularity)
                
                dl_pred = np.clip(np.expm1(dl_pred_log), 
                               st.session_state.min_popularity, 
                               st.session_state.max_popularity)

                # Display results
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Ridge Prediction", 
                        f"{ridge_pred:.1f}",
                        help=f"Log-space prediction: {ridge_pred_log:.2f}"
                    )
                
                with col2:
                    st.metric(
                        "DL Prediction", 
                        f"{dl_pred:.1f}",
                        help=f"Log-space prediction: {dl_pred_log:.2f}"
                    )
                
                # Show prediction explanation
                st.info(f"""
                **Interpretation:**
                - Popularity scores typically range from {st.session_state.min_popularity:.1f} to {st.session_state.max_popularity:.1f}
                - Higher values indicate more popular movies
                - The DL model may better capture complex patterns
                - The Ridge model provides more conservative estimates
                """)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.error("Please check your inputs and try again")

    elif 'models_trained' in st.session_state and not st.session_state.models_trained:
        st.warning("Please train the models first by clicking the 'Train and Evaluate Models' button")





