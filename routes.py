from flask import render_template, request, redirect, url_for, flash, session, jsonify
from app import app
from extensions import db
from models import User, Transaction, Budget, PredictionHistory
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
from calendar import monthrange
import calendar
from ml_service import MLForecastService
from sqlalchemy import func, extract, desc

# Predefined categories
EXPENSE_CATEGORIES = [
    'Food & Dining', 'Transportation', 'Shopping', 'Entertainment',
    'Bills & Utilities', 'Healthcare', 'Education', 'Travel',
    'Groceries', 'Rent/Mortgage', 'Insurance', 'Miscellaneous'
]

INCOME_CATEGORIES = [
    'Salary', 'Freelance', 'Business', 'Investment',
    'Rental Income', 'Bonus', 'Gift', 'Other Income'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    # Monthly data
    monthly_income = db.session.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == user_id,
        Transaction.transaction_type == 'income',
        extract('month', Transaction.date) == current_month,
        extract('year', Transaction.date) == current_year
    ).scalar() or 0
    
    monthly_expenses = db.session.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == user_id,
        Transaction.transaction_type == 'expense',
        extract('month', Transaction.date) == current_month,
        extract('year', Transaction.date) == current_year
    ).scalar() or 0
    
    # Transaction statistics
    total_transactions = Transaction.query.filter(
        Transaction.user_id == user_id,
        extract('month', Transaction.date) == current_month,
        extract('year', Transaction.date) == current_year
    ).count()
    
    avg_transaction = db.session.query(func.avg(Transaction.amount)).filter(
        Transaction.user_id == user_id,
        Transaction.transaction_type == 'expense',
        extract('month', Transaction.date) == current_month,
        extract('year', Transaction.date) == current_year
    ).scalar() or 0
    
    # Spending velocity (daily average)
    days_in_month = monthrange(current_year, current_month)[1]
    current_day = datetime.now().day
    spending_velocity = monthly_expenses / current_day if current_day > 0 else 0
    
    # Category expenses for current month
    category_expenses = db.session.query(
        Transaction.category,
        func.sum(Transaction.amount).label('total')
    ).filter(
        Transaction.user_id == user_id,
        Transaction.transaction_type == 'expense',
        extract('month', Transaction.date) == current_month,
        extract('year', Transaction.date) == current_year
    ).group_by(Transaction.category).all()
    
    category_data = {expense.category: float(expense.total) for expense in category_expenses}
    
    # Monthly trends (last 12 months)
    monthly_trends = []
    for i in range(12):
        date_obj = datetime.now() - timedelta(days=30*i)
        month = date_obj.month
        year = date_obj.year
        
        income = db.session.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            Transaction.transaction_type == 'income',
            extract('month', Transaction.date) == month,
            extract('year', Transaction.date) == year
        ).scalar() or 0
        
        expenses = db.session.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            Transaction.transaction_type == 'expense',
            extract('month', Transaction.date) == month,
            extract('year', Transaction.date) == year
        ).scalar() or 0
        
        monthly_trends.append({
            'month': calendar.month_name[month],
            'short_month': calendar.month_abbr[month],
            'year': year,
            'income': float(income),
            'expenses': float(expenses),
            'savings': float(income - expenses)
        })
    
    monthly_trends.reverse()
    
    # Category trends (last 6 months)
    category_trends = {}
    for category in EXPENSE_CATEGORIES:
        trend_data = []
        for i in range(6):
            date_obj = datetime.now() - timedelta(days=30*i)
            month = date_obj.month
            year = date_obj.year
            
            amount = db.session.query(func.sum(Transaction.amount)).filter(
                Transaction.user_id == user_id,
                Transaction.transaction_type == 'expense',
                Transaction.category == category,
                extract('month', Transaction.date) == month,
                extract('year', Transaction.date) == year
            ).scalar() or 0
            
            trend_data.append(float(amount))
        
        trend_data.reverse()
        category_trends[category] = trend_data
    
    # Recent transactions
    recent_transactions = Transaction.query.filter_by(user_id=user_id).order_by(
        desc(Transaction.date), desc(Transaction.created_at)
    ).limit(5).all()
    
    return render_template('dashboard.html',
                         monthly_income=monthly_income,
                         monthly_expenses=monthly_expenses,
                         total_transactions=total_transactions,
                         avg_transaction_amount=avg_transaction,
                         spending_velocity=spending_velocity,
                         category_expenses=category_data,
                         monthly_trends=monthly_trends,
                         category_trends=category_trends,
                         recent_transactions=recent_transactions,
                         current_month=calendar.month_name[current_month])

@app.route('/transactions')
@login_required
def transactions():
    user_id = session['user_id']
    page = request.args.get('page', 1, type=int)
    
    transactions = Transaction.query.filter_by(user_id=user_id).order_by(
        desc(Transaction.date), desc(Transaction.created_at)
    ).paginate(page=page, per_page=20, error_out=False)
    
    return render_template('transactions.html',
                         transactions=transactions,
                         expense_categories=EXPENSE_CATEGORIES,
                         income_categories=INCOME_CATEGORIES)

@app.route('/add_transaction', methods=['POST'])
@login_required
def add_transaction():
    user_id = session['user_id']
    
    transaction = Transaction(
        user_id=user_id,
        amount=float(request.form['amount']),
        description=request.form.get('description', ''),
        category=request.form['category'],
        transaction_type=request.form['transaction_type'],
        date=datetime.strptime(request.form['date'], '%Y-%m-%d').date()
    )
    
    db.session.add(transaction)
    db.session.commit()
    
    flash('Transaction added successfully!', 'success')
    return redirect(url_for('transactions'))

@app.route('/delete_transaction/<int:transaction_id>')
@login_required
def delete_transaction(transaction_id):
    user_id = session['user_id']
    transaction = Transaction.query.filter_by(id=transaction_id, user_id=user_id).first()
    
    if transaction:
        db.session.delete(transaction)
        db.session.commit()
        flash('Transaction deleted successfully!', 'success')
    else:
        flash('Transaction not found!', 'error')
    
    return redirect(url_for('transactions'))

@app.route('/budget')
@login_required
def budget():
    user_id = session['user_id']
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    # Get current month budgets
    budgets = Budget.query.filter_by(
        user_id=user_id,
        month=current_month,
        year=current_year
    ).all()
    
    budget_data = []
    total_budget = 0
    total_spent = 0
    
    for budget in budgets:
        # Calculate spent amount for this category
        spent = db.session.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == user_id,
            Transaction.transaction_type == 'expense',
            Transaction.category == budget.category,
            extract('month', Transaction.date) == current_month,
            extract('year', Transaction.date) == current_year
        ).scalar() or 0
        
        percentage = (spent / budget.amount * 100) if budget.amount > 0 else 0
        remaining = budget.amount - spent
        
        budget_data.append({
            'category': budget.category,
            'budget': budget.amount,
            'spent': spent,
            'remaining': remaining,
            'percentage': min(percentage, 100)
        })
        
        total_budget += budget.amount
        total_spent += spent
    
    return render_template('budget.html',
                         budget_data=budget_data,
                         total_budget=total_budget,
                         total_spent=total_spent,
                         expense_categories=EXPENSE_CATEGORIES,
                         current_month=calendar.month_name[current_month])

@app.route('/add_budget', methods=['POST'])
@login_required
def add_budget():
    user_id = session['user_id']
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    # Check if budget already exists for this category and month
    existing_budget = Budget.query.filter_by(
        user_id=user_id,
        category=request.form['category'],
        month=current_month,
        year=current_year
    ).first()
    
    if existing_budget:
        existing_budget.amount = float(request.form['amount'])
        flash('Budget updated successfully!', 'success')
    else:
        budget = Budget(
            user_id=user_id,
            category=request.form['category'],
            amount=float(request.form['amount']),
            month=current_month,
            year=current_year
        )
        db.session.add(budget)
        flash('Budget set successfully!', 'success')
    
    db.session.commit()
    return redirect(url_for('budget'))

@app.route('/forecast')
@login_required
def forecast():
    user_id = session['user_id']
    
    # Get month and year from query parameters for historical viewing
    selected_month = request.args.get('month', datetime.now().month, type=int)
    selected_year = request.args.get('year', datetime.now().year, type=int)
    view_type = request.args.get('view', 'current')  # 'current', 'future', 'historical'
    
    ml_service = MLForecastService(user_id)
    
    if view_type == 'historical':
        # Show historical data for selected month
        predictions = ml_service.get_historical_data(selected_month, selected_year)
        insights = ml_service.generate_historical_insights(selected_month, selected_year)
        tips = ml_service.get_financial_tips_for_period(selected_month, selected_year)
    elif view_type == 'future':
        # Show future predictions
        months_ahead = request.args.get('months', 1, type=int)
        predictions = ml_service.predict_future_expenses(months_ahead)
        insights = ml_service.generate_future_insights(months_ahead)
        tips = ml_service.get_predictive_tips(months_ahead)
    else:
        # Default: current month predictions
        predictions = ml_service.predict_next_month()
        insights = ml_service.generate_insights()
        tips = ml_service.get_financial_tips()
    
    # Available months for selection (last 24 months + next 12 months)
    available_dates = []
    current_date = datetime.now()
    
    # Add past months
    for i in range(24, 0, -1):
        date_obj = current_date - timedelta(days=30*i)
        available_dates.append({
            'month': date_obj.month,
            'year': date_obj.year,
            'name': f"{calendar.month_name[date_obj.month]} {date_obj.year}",
            'type': 'historical'
        })
    
    # Add current month
    available_dates.append({
        'month': current_date.month,
        'year': current_date.year,
        'name': f"{calendar.month_name[current_date.month]} {current_date.year}",
        'type': 'current'
    })
    
    # Add future months
    for i in range(1, 13):
        date_obj = current_date + timedelta(days=30*i)
        available_dates.append({
            'month': date_obj.month,
            'year': date_obj.year,
            'name': f"{calendar.month_name[date_obj.month]} {date_obj.year}",
            'type': 'future'
        })
    
    return render_template('forecast.html',
                         predictions=predictions,
                         insights=insights,
                         tips=tips,
                         available_dates=available_dates,
                         selected_month=selected_month,
                         selected_year=selected_year,
                         view_type=view_type,
                         current_month_name=calendar.month_name[selected_month])

@app.route('/api/forecast/accuracy')
@login_required
def forecast_accuracy():
    user_id = session['user_id']
    ml_service = MLForecastService(user_id)
    accuracy_data = ml_service.get_prediction_accuracy()
    return jsonify(accuracy_data)

@app.route('/api/seasonal-analysis')
@login_required
def seasonal_analysis():
    user_id = session['user_id']
    ml_service = MLForecastService(user_id)
    seasonal_data = ml_service.get_seasonal_analysis()
    return jsonify(seasonal_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
