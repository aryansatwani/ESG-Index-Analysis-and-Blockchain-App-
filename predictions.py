from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

class ESGPredictor:
    def __init__(self, forecast_months=6):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.forecast_months = forecast_months

    def get_filters(self, data):
        """Get available filter options from data"""
        df = pd.DataFrame(data)
        filters = {
            'companies': sorted(df['company'].unique().tolist()),
            'locations': sorted(df['location'].unique().tolist()),
            'company_locations': sorted(df.apply(lambda x: f"{x['company']} - {x['location']}", axis=1).unique().tolist())
        }
        return filters

    def filter_data(self, data, company=None, location=None):
        """Filter data based on company and/or location"""
        df = pd.DataFrame(data)
        
        if company and location:
            filtered_df = df[(df['company'] == company) & (df['location'] == location)]
        elif company:
            filtered_df = df[df['company'] == company]
        elif location:
            filtered_df = df[df['location'] == location]
        else:
            filtered_df = df

        return filtered_df.to_dict('records')

    def prepare_data(self, data):
        """Prepare data for prediction"""
        df = pd.DataFrame(data)
        
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Create time-based features
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Add rolling averages for numeric columns
        numeric_cols = ['carbon_emissions', 'diversity', 'safety', 'compliance', 'energy_efficiency']
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_rolling_avg'] = df[col].rolling(window=2, min_periods=1).mean()

        return df

    def generate_future_features(self, last_date, n_months):
        """Generate feature values for future dates"""
        future_dates = [last_date + relativedelta(months=i+1) for i in range(n_months)]
        future_features = pd.DataFrame({
            'timestamp': future_dates,
            'year': [d.year for d in future_dates],
            'month': [d.month for d in future_dates],
            'day_of_year': [d.timetuple().tm_yday for d in future_dates]
        })
        return future_features

    def train_predict(self, data, target_metric, filter_type='all', filter_value=None):
        """
        Train model and make predictions
        filter_type: 'all', 'company', 'location', or 'company_location'
        """
        if len(data) < 2:
            return self._generate_empty_prediction()

        # Filter data if needed
        if filter_type == 'company':
            company = filter_value
            data = self.filter_data(data, company=company)
        elif filter_type == 'location':
            location = filter_value
            data = self.filter_data(data, location=location)
        elif filter_type == 'company_location':
            company, location = filter_value.split(' - ')
            data = self.filter_data(data, company=company, location=location)

        df = self.prepare_data(data)
        
        if len(df) < 2:
            return self._generate_empty_prediction()

        # Prepare features
        feature_cols = ['year', 'month', 'day_of_year']
        numeric_cols = ['carbon_emissions', 'diversity', 'safety', 'compliance', 'energy_efficiency']
        
        # Add rolling average features
        for col in numeric_cols:
            if col != target_metric and col in df.columns:
                feature_cols.append(f'{col}_rolling_avg')

        X = df[feature_cols].values
        y = df[target_metric].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Generate future dates and features
        last_date = df['timestamp'].max()
        future_features = self.generate_future_features(last_date, self.forecast_months)
        
        # Prepare future features for prediction
        future_X = future_features[['year', 'month', 'day_of_year']].values
        
        # Add dummy values for rolling averages
        if len(feature_cols) > 3:  # if we have rolling average features
            dummy_values = np.zeros((len(future_X), len(feature_cols) - 3))
            future_X = np.concatenate([future_X, dummy_values], axis=1)
        
        # Scale future features
        future_X_scaled = self.scaler.transform(future_X)
        
        # Make predictions
        future_predictions = self.model.predict(future_X_scaled)
        
        # Calculate confidence intervals
        prediction_intervals = self._calculate_confidence_intervals(future_predictions)
        
        # Get historical values
        historical_values = df[[target_metric, 'timestamp']].to_dict('records')
        
        return {
            'current_date': last_date.strftime('%Y-%m-%d'),
            'forecast_dates': future_features['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
            'historical_values': historical_values,
            'current_value': float(y[-1]),
            'forecasted_values': [float(v) for v in future_predictions],
            'confidence_intervals': prediction_intervals,
            'trend': 'Improving' if future_predictions[-1] > y[-1] else 'Declining',
            'change_percent': float(((future_predictions[-1] - y[-1]) / y[-1]) * 100)
        }

    def _generate_empty_prediction(self):
        """Generate empty prediction structure when insufficient data"""
        return {
            'current_date': datetime.now().strftime('%Y-%m-%d'),
            'forecast_dates': [],
            'historical_values': [],
            'current_value': 0,
            'forecasted_values': [],
            'confidence_intervals': {'lower': [], 'upper': []},
            'trend': 'Insufficient data',
            'change_percent': 0
        }

    def _calculate_confidence_intervals(self, predictions, confidence=0.95):
        """Calculate confidence intervals for predictions"""
        intervals = {
            'lower': [],
            'upper': []
        }
        
        for pred in predictions:
            # Generate bootstrap samples
            bootstrap_samples = np.random.normal(pred, abs(pred * 0.1), 1000)
            
            # Calculate confidence intervals
            intervals['lower'].append(float(np.percentile(bootstrap_samples, 
                                                        (1 - confidence) * 100 / 2)))
            intervals['upper'].append(float(np.percentile(bootstrap_samples, 
                                                        (1 + confidence) * 100 / 2)))
            
        return intervals