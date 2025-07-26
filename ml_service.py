import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from calendar import monthrange
import calendar
import json
from sqlalchemy import func, extract
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from extensions import db
from models import Transaction, PredictionHistory

class MLForecastService:
    def __init__(self, user_id):
        self.user_id = user_id
        
    def get_transaction_data(self):
        """Fetch user's transaction data"""
        transactions = db.session.query(
            Transaction.date,
            Transaction.amount,
            Transaction.category,
            Transaction.transaction_type
        ).filter(
            Transaction.user_id == self.user_id
        ).order_by(Transaction.date).all()
        
        return pd.DataFrame([{
            'date': t.date,
            'amount': t.amount,
            'category': t.category,
            'type': t.transaction_type
        } for t in transactions])
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        if df.empty:
            return pd.DataFrame()
        
        # Create monthly aggregations
        df['year_month'] = df['date'].apply(lambda x: f"{x.year}-{x.month:02d}")
        
        monthly_data = []
        
        for year_month in df['year_month'].unique():
            month_data = df[df['year_month'] == year_month]
            
            year, month = map(int, year_month.split('-'))
            
            # Calculate features
            total_expenses = month_data[month_data['type'] == 'expense']['amount'].sum()
            total_income = month_data[month_data['type'] == 'income']['amount'].sum()
            transaction_count = len(month_data)
            avg_expense = month_data[month_data['type'] == 'expense']['amount'].mean() if len(month_data[month_data['type'] == 'expense']) > 0 else 0
            
            # Seasonal features
            quarter = (month - 1) // 3 + 1
            is_holiday_season = 1 if month in [11, 12, 1] else 0
            is_summer = 1 if month in [6, 7, 8] else 0
            
            # Category diversity
            unique_categories = month_data['category'].nunique()
            
            # Days in month
            days_in_month = monthrange(year, month)[1]
            
            monthly_data.append({
                'year': year,
                'month': month,
                'year_month': year_month,
                'total_expenses': total_expenses,
                'total_income': total_income,
                'net_savings': total_income - total_expenses,
                'transaction_count': transaction_count,
                'avg_expense': avg_expense,
                'quarter': quarter,
                'is_holiday_season': is_holiday_season,
                'is_summer': is_summer,
                'unique_categories': unique_categories,
                'days_in_month': days_in_month,
                'expense_per_day': total_expenses / days_in_month
            })
        
        return pd.DataFrame(monthly_data)
    
    def get_category_features(self, df, target_month, target_year):
        """Get category-wise spending patterns"""
        if df.empty:
            return {}
        
        # Get historical category spending
        expense_data = df[df['type'] == 'expense'].copy()
        expense_data['year_month'] = expense_data['date'].apply(lambda x: f"{x.year}-{x.month:02d}")
        
        category_patterns = {}
        
        for category in expense_data['category'].unique():
            cat_data = expense_data[expense_data['category'] == category]
            monthly_spending = cat_data.groupby('year_month')['amount'].sum()
            
            if len(monthly_spending) >= 2:
                # Calculate trend
                trend = np.polyfit(range(len(monthly_spending)), monthly_spending.values, 1)[0]
                avg_spending = monthly_spending.mean()
                
                # Seasonal factor
                seasonal_factors = {}
                for month in range(1, 13):
                    month_data = cat_data[cat_data['date'].apply(lambda x: x.month) == month]
                    if not month_data.empty:
                        seasonal_factors[month] = month_data['amount'].sum() / len(month_data['date'].apply(lambda x: x.year).unique())
                
                category_patterns[category] = {
                    'avg_spending': avg_spending,
                    'trend': trend,
                    'seasonal_factor': seasonal_factors.get(target_month, avg_spending)
                }
        
        return category_patterns
    
    def train_models(self, features_df):
        """Train multiple ML models"""
        if len(features_df) < 3:
            return None, None, None
        
        # Prepare features
        X = features_df[[
            'month', 'quarter', 'is_holiday_season', 'is_summer',
            'total_income', 'transaction_count', 'unique_categories',
            'days_in_month'
        ]].fillna(0)
        
        y = features_df['total_expenses'].fillna(0)
        
        if len(X) < 3:
            return None, None, None
        
        models = {}
        scores = {}
        
        try:
            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
            rf_model.fit(X, y)
            rf_pred = rf_model.predict(X)
            models['random_forest'] = rf_model
            scores['random_forest'] = r2_score(y, rf_pred)
            
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            lr_pred = lr_model.predict(X)
            models['linear'] = lr_model
            scores['linear'] = r2_score(y, lr_pred)
            
            # Polynomial Features
            if len(X) >= 5:
                poly_features = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = poly_features.fit_transform(X)
                poly_model = LinearRegression()
                poly_model.fit(X_poly, y)
                poly_pred = poly_model.predict(X_poly)
                models['polynomial'] = (poly_model, poly_features)
                scores['polynomial'] = r2_score(y, poly_pred)
            
        except Exception as e:
            print(f"Model training error: {e}")
            return None, None, None
        
        # Return best model
        if scores:
            best_model_name = max(scores, key=scores.get)
            return models[best_model_name], best_model_name, scores[best_model_name]
        
        return None, None, None
    
    def predict_next_month(self):
        """Predict expenses for next month"""
        df = self.get_transaction_data()
        
        if df.empty:
            return self._get_empty_prediction()
        
        # Get next month details
        next_month_date = datetime.now() + timedelta(days=32)
        next_month = next_month_date.month
        next_year = next_month_date.year
        
        return self._generate_prediction(df, next_month, next_year)
    
    def predict_future_expenses(self, months_ahead=1):
        """Predict expenses for multiple months ahead"""
        df = self.get_transaction_data()
        
        if df.empty:
            return self._get_empty_prediction()
        
        # Calculate target month
        target_date = datetime.now() + timedelta(days=30 * months_ahead)
        target_month = target_date.month
        target_year = target_date.year
        
        return self._generate_prediction(df, target_month, target_year, months_ahead)
    
    def get_historical_data(self, month, year):
        """Get historical data for a specific month"""
        # Get actual data for the month
        actual_expenses = db.session.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == self.user_id,
            Transaction.transaction_type == 'expense',
            extract('month', Transaction.date) == month,
            extract('year', Transaction.date) == year
        ).scalar() or 0
        
        # Get category breakdown
        category_data = db.session.query(
            Transaction.category,
            func.sum(Transaction.amount).label('total')
        ).filter(
            Transaction.user_id == self.user_id,
            Transaction.transaction_type == 'expense',
            extract('month', Transaction.date) == month,
            extract('year', Transaction.date) == year
        ).group_by(Transaction.category).all()
        
        category_breakdown = {cat.category: float(cat.total) for cat in category_data}
        
        return {
            'total_predicted': actual_expenses,
            'prediction_month': f"{calendar.month_name[month]} {year}",
            'category_breakdown': category_breakdown,
            'confidence': 'Historical Data',
            'is_historical': True
        }
    
    def _generate_prediction(self, df, target_month, target_year, months_ahead=1):
        """Generate prediction for specific month"""
        features_df = self.prepare_features(df)
        
        if features_df.empty or len(features_df) < 2:
            return self._get_empty_prediction()
        
        # Train models
        model, model_name, score = self.train_models(features_df)
        
        if model is None:
            return self._get_fallback_prediction(df, target_month, target_year)
        
        # Prepare prediction features
        recent_income = features_df['total_income'].tail(3).mean()
        recent_transactions = features_df['transaction_count'].tail(3).mean()
        recent_categories = features_df['unique_categories'].tail(3).mean()
        
        quarter = (target_month - 1) // 3 + 1
        is_holiday_season = 1 if target_month in [11, 12, 1] else 0
        is_summer = 1 if target_month in [6, 7, 8] else 0
        days_in_month = monthrange(target_year, target_month)[1]
        
        X_pred = np.array([[
            target_month, quarter, is_holiday_season, is_summer,
            recent_income, recent_transactions, recent_categories, days_in_month
        ]])
        
        # Make prediction
        if model_name == 'polynomial':
            model, poly_features = model
            X_pred = poly_features.transform(X_pred)
        
        predicted_total = model.predict(X_pred)[0]
        predicted_total = max(0, predicted_total)  # Ensure non-negative
        
        # Category-wise prediction
        category_patterns = self.get_category_features(df, target_month, target_year)
        category_breakdown = self._predict_category_breakdown(predicted_total, category_patterns)
        
        # Confidence score
        confidence = self._calculate_confidence(score, len(features_df), months_ahead)
        
        # Save prediction
        self._save_prediction(target_month, target_year, predicted_total, category_breakdown, model_name)
        
        return {
            'total_predicted': round(predicted_total, 2),
            'prediction_month': f"{calendar.month_name[target_month]} {target_year}",
            'category_breakdown': category_breakdown,
            'confidence': confidence,
            'model_used': model_name,
            'accuracy_score': round(score * 100, 1) if score else 0
        }
    
    def _predict_category_breakdown(self, total_amount, category_patterns):
        """Predict category-wise breakdown"""
        if not category_patterns:
            # Fallback to even distribution among common categories
            common_categories = ['Food & Dining', 'Transportation', 'Bills & Utilities', 'Shopping']
            amount_per_category = total_amount / len(common_categories)
            return {cat: round(amount_per_category, 2) for cat in common_categories}
        
        # Calculate weights based on historical patterns
        total_weight = sum(pattern['avg_spending'] for pattern in category_patterns.values())
        
        if total_weight == 0:
            return {cat: 0 for cat in category_patterns.keys()}
        
        breakdown = {}
        for category, pattern in category_patterns.items():
            weight = pattern['avg_spending'] / total_weight
            predicted_amount = total_amount * weight
            
            # Apply trend and seasonal adjustments
            trend_factor = 1 + (pattern['trend'] / pattern['avg_spending']) if pattern['avg_spending'] > 0 else 1
            seasonal_factor = pattern['seasonal_factor'] / pattern['avg_spending'] if pattern['avg_spending'] > 0 else 1
            
            adjusted_amount = predicted_amount * trend_factor * seasonal_factor
            breakdown[category] = round(max(0, adjusted_amount), 2)
        
        return breakdown
    
    def _calculate_confidence(self, model_score, data_points, months_ahead):
        """Calculate confidence level"""
        if model_score is None:
            return "Low"
        
        base_confidence = model_score * 100
        
        # Adjust for data points
        if data_points < 3:
            base_confidence *= 0.6
        elif data_points < 6:
            base_confidence *= 0.8
        
        # Adjust for prediction distance
        base_confidence *= (0.95 ** months_ahead)
        
        if base_confidence >= 75:
            return "High"
        elif base_confidence >= 50:
            return "Medium"
        else:
            return "Low"
    
    def _get_empty_prediction(self):
        """Return empty prediction when no data available"""
        next_month = datetime.now() + timedelta(days=32)
        return {
            'total_predicted': 0,
            'prediction_month': f"{calendar.month_name[next_month.month]} {next_month.year}",
            'category_breakdown': {},
            'confidence': 'No Data',
            'message': 'Add more transactions to get predictions'
        }
    
    def _get_fallback_prediction(self, df, target_month, target_year):
        """Fallback prediction using simple averages"""
        recent_expenses = df[df['type'] == 'expense']['amount'].tail(10).sum()
        avg_monthly = recent_expenses / 3 if len(df) > 0 else 0
        
        return {
            'total_predicted': round(avg_monthly, 2),
            'prediction_month': f"{calendar.month_name[target_month]} {target_year}",
            'category_breakdown': {'Estimated': round(avg_monthly, 2)},
            'confidence': 'Low',
            'model_used': 'simple_average'
        }
    
    def _save_prediction(self, month, year, amount, category_breakdown, model_name):
        """Save prediction to database"""
        try:
            prediction = PredictionHistory(
                user_id=self.user_id,
                prediction_month=month,
                prediction_year=year,
                predicted_amount=amount,
                model_used=model_name,
                category_predictions=json.dumps(category_breakdown)
            )
            db.session.add(prediction)
            db.session.commit()
        except Exception as e:
            print(f"Error saving prediction: {e}")
            db.session.rollback()
    
    def generate_insights(self):
        """Generate AI insights for spending patterns"""
        df = self.get_transaction_data()
        
        if df.empty:
            return ["Add some transactions to get personalized insights!"]
        
        insights = []
        
        # Spending trend analysis
        expense_data = df[df['type'] == 'expense'].copy()
        if not expense_data.empty:
            expense_data['month'] = expense_data['date'].apply(lambda x: x.strftime('%Y-%m'))
            monthly_expenses = expense_data.groupby('month')['amount'].sum()
            
            if len(monthly_expenses) >= 2:
                recent_avg = monthly_expenses.tail(3).mean()
                older_avg = monthly_expenses.head(-3).mean() if len(monthly_expenses) > 3 else recent_avg
                
                if recent_avg > older_avg * 1.1:
                    insights.append("📈 Your spending has increased by more than 10% recently. Consider reviewing your budget.")
                elif recent_avg < older_avg * 0.9:
                    insights.append("📉 Great job! Your spending has decreased by more than 10% recently.")
                else:
                    insights.append("📊 Your spending patterns have been relatively stable.")
        
        # Category analysis
        category_spending = expense_data.groupby('category')['amount'].sum().sort_values(ascending=False)
        if not category_spending.empty:
            top_category = category_spending.index[0]
            top_amount = category_spending.iloc[0]
            total_spending = category_spending.sum()
            percentage = (top_amount / total_spending) * 100 if total_spending > 0 else 0
            
            insights.append(f"🏆 Your highest spending category is '{top_category}' at {percentage:.1f}% of total expenses.")
            
            if percentage > 40:
                insights.append(f"⚠️ Consider diversifying your expenses - '{top_category}' dominates your spending.")
        
        # Frequency analysis
        if len(expense_data) > 0:
            avg_transaction = expense_data['amount'].mean()
            recent_transactions = expense_data.tail(10)['amount'].mean()
            
            if recent_transactions > avg_transaction * 1.2:
                insights.append("💳 Your recent transactions are larger than usual. Monitor for any unusual spending.")
            elif recent_transactions < avg_transaction * 0.8:
                insights.append("✅ Your recent transactions are smaller than usual - good spending control!")
        
        return insights[:5]  # Return top 5 insights
    
    def generate_historical_insights(self, month, year):
        """Generate insights for historical period"""
        insights = []
        
        # Get data for the month
        month_expenses = db.session.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == self.user_id,
            Transaction.transaction_type == 'expense',
            extract('month', Transaction.date) == month,
            extract('year', Transaction.date) == year
        ).scalar() or 0
        
        # Compare with other months
        avg_expenses = db.session.query(func.avg(
            db.session.query(func.sum(Transaction.amount)).filter(
                Transaction.user_id == self.user_id,
                Transaction.transaction_type == 'expense'
            ).group_by(
                extract('year', Transaction.date),
                extract('month', Transaction.date)
            ).subquery().c.sum
        )).scalar() or 0
        
        if month_expenses > avg_expenses * 1.2:
            insights.append(f"📊 {calendar.month_name[month]} {year} had higher than average spending (+{((month_expenses/avg_expenses-1)*100):.1f}%)")
        elif month_expenses < avg_expenses * 0.8:
            insights.append(f"📊 {calendar.month_name[month]} {year} had lower than average spending (-{((1-month_expenses/avg_expenses)*100):.1f}%)")
        
        return insights
    
    def generate_future_insights(self, months_ahead):
        """Generate insights for future predictions"""
        insights = []
        
        target_date = datetime.now() + timedelta(days=30 * months_ahead)
        target_month = target_date.month
        
        # Seasonal insights
        if target_month in [11, 12]:
            insights.append("🎄 Holiday season ahead - expect higher spending on gifts and celebrations.")
        elif target_month in [6, 7, 8]:
            insights.append("☀️ Summer months often see increased travel and entertainment expenses.")
        elif target_month in [1, 2]:
            insights.append("💪 New Year is a great time to set and stick to financial resolutions.")
        
        if months_ahead > 3:
            insights.append("📅 Long-term predictions have higher uncertainty - regularly update your data for better accuracy.")
        
        return insights
    
    def get_financial_tips(self):
        """Get personalized financial tips"""
        df = self.get_transaction_data()
        
        tips = [
            "💡 Track every expense, no matter how small - they add up quickly!",
            "🎯 Set specific, measurable financial goals for better motivation.",
            "📱 Use the 50/30/20 rule: 50% needs, 30% wants, 20% savings.",
            "💰 Automate your savings to ensure consistent progress.",
            "🔍 Review and categorize expenses weekly to stay on track."
        ]
        
        if not df.empty:
            expense_data = df[df['type'] == 'expense']
            if not expense_data.empty:
                # Check for frequent small expenses
                small_expenses = expense_data[expense_data['amount'] < 50]
                if len(small_expenses) > len(expense_data) * 0.6:
                    tips.insert(0, "🎯 You have many small expenses - consider bundling purchases to reduce impulse buying.")
                
                # Check for category concentration
                category_counts = expense_data['category'].value_counts()
                if len(category_counts) > 0 and category_counts.iloc[0] > len(expense_data) * 0.5:
                    top_category = category_counts.index[0]
                    tips.insert(0, f"⚡ Most of your spending is in '{top_category}' - look for ways to optimize this category.")
        
        return tips[:4]
    
    def get_financial_tips_for_period(self, month, year):
        """Get tips specific to a historical period"""
        tips = []
        
        # Season-specific tips
        if month in [11, 12]:
            tips.append("🎁 Holiday spending tip: Set a gift budget before shopping and stick to it.")
        elif month in [6, 7, 8]:
            tips.append("🏖️ Summer saving tip: Look for free outdoor activities instead of expensive entertainment.")
        elif month in [1, 2]:
            tips.append("💪 New Year tip: Start the year with a clear budget plan and financial goals.")
        
        tips.extend([
            "📊 Review this month's patterns to improve future spending decisions.",
            "🔄 Use historical data to identify recurring expenses and plan ahead.",
            "📈 Compare different months to understand your spending seasonality."
        ])
        
        return tips[:4]
    
    def get_predictive_tips(self, months_ahead):
        """Get tips for future financial planning"""
        tips = []
        
        if months_ahead >= 6:
            tips.append("📅 Long-term planning: Consider seasonal variations in your budget.")
            tips.append("💡 Start building an emergency fund if you haven't already.")
        
        tips.extend([
            "🎯 Use predictions to set realistic spending targets.",
            "📊 Regular data updates improve prediction accuracy.",
            "⚡ Plan for unexpected expenses by adding a 10-15% buffer to predictions."
        ])
        
        return tips[:4]
    
    def get_prediction_accuracy(self):
        """Calculate prediction accuracy from historical data"""
        predictions = PredictionHistory.query.filter_by(user_id=self.user_id).all()
        
        accuracy_data = {
            'total_predictions': len(predictions),
            'accurate_predictions': 0,
            'avg_accuracy': 0,
            'by_model': {}
        }
        
        for pred in predictions:
            if pred.actual_amount is not None:
                error = abs(pred.predicted_amount - pred.actual_amount)
                accuracy = max(0, 100 - (error / pred.predicted_amount * 100)) if pred.predicted_amount > 0 else 0
                
                accuracy_data['accurate_predictions'] += 1
                accuracy_data['avg_accuracy'] += accuracy
                
                if pred.model_used not in accuracy_data['by_model']:
                    accuracy_data['by_model'][pred.model_used] = []
                accuracy_data['by_model'][pred.model_used].append(accuracy)
        
        if accuracy_data['accurate_predictions'] > 0:
            accuracy_data['avg_accuracy'] /= accuracy_data['accurate_predictions']
        
        return accuracy_data
    
    def get_seasonal_analysis(self):
        """Analyze seasonal spending patterns"""
        df = self.get_transaction_data()
        
        if df.empty:
            return {}
        
        expense_data = df[df['type'] == 'expense'].copy()
        expense_data['month'] = expense_data['date'].apply(lambda x: x.month)
        
        seasonal_data = {}
        for month in range(1, 13):
            month_data = expense_data[expense_data['month'] == month]
            if not month_data.empty:
                seasonal_data[calendar.month_name[month]] = {
                    'avg_spending': month_data['amount'].mean(),
                    'total_spending': month_data['amount'].sum(),
                    'transaction_count': len(month_data)
                }
        
        return seasonal_data
     
