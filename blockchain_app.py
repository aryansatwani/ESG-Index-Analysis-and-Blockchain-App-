from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from models_esg import db, init_db, ESGRecord, User
from config_esg import Config
from blockchain import Blockchain
from email.mime.text import MIMEText
import smtplib
import pandas as pd
import io
from io import BytesIO
import xlsxwriter
from datetime import datetime
from werkzeug.utils import secure_filename
import os
import numpy as np

app = Flask(__name__)
app.config.from_object(Config)

init_db(app)
blockchain = Blockchain()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes

@app.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of records per page
    
    # Get search parameters
    company = request.args.get('company', '').lower()
    location = request.args.get('location', '').lower()
    rating = request.args.get('rating', '')

    # Filter the blockchain data
    filtered_blocks = []
    for block in blockchain.chain:
        filtered_data = []
        for record in block['data']:
            if (company == '' or company in record['company'].lower()) and \
               (location == '' or location in record['location'].lower()) and \
               (rating == '' or rating == record.get('esg_rating', '')):
                filtered_data.append(record)
        
        if filtered_data:
            filtered_block = block.copy()
            filtered_block['data'] = filtered_data
            filtered_blocks.append(filtered_block)

    # Paginate the filtered results
    total_blocks = len(filtered_blocks)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_blocks = filtered_blocks[start_idx:end_idx]

    # Create pagination object
    class Pagination:
        def __init__(self, items, page, per_page, total):
            self.items = items
            self.page = page
            self.per_page = per_page
            self.total = total
            self.pages = (total + per_page - 1) // per_page
            self.has_prev = page > 1
            self.has_next = page < self.pages
            self.prev_num = page - 1
            self.next_num = page + 1

    pagination = Pagination(paginated_blocks, page, per_page, total_blocks)
    
    return render_template('home.html', records=pagination)



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('add_data'))
        else:
            flash("Invalid username or password")
    return render_template('login.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
                # Generate a reset link (mock example)
            reset_link = url_for('reset_password', user_id=user.id, _external=True)
            
                        # Send email
            send_reset_email(email, reset_link)
            flash("Password reset link sent to your email!")
        else:
            flash("Email not found.")
    return render_template('forgot_password.html')
            
def send_reset_email(to_email, reset_link):
    sender_email = "your_email@example.com"
    sender_password = "your_email_password"
            
    msg = MIMEText(f"Click the link to reset your password: {reset_link}")
    msg['Subject'] = "Password Reset Request"
    msg['From'] = sender_email
    msg['To'] = to_email
            
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)
        
@app.route('/reset_password/<int:user_id>', methods=['GET', 'POST'])
def reset_password(user_id):
    user = User.query.get(user_id)
    if not user:
        flash("Invalid reset link.")
        return redirect(url_for('index'))

    if request.method == 'POST':
        new_password = request.form['password']
        confirm_password = request.form['confirm_password']

        if new_password != confirm_password:
            flash("Passwords do not match!")
        else:
            user.password = new_password
            db.session.commit()
            flash("Password reset successful! Please log in.")
            return redirect(url_for('login'))

    return render_template('reset_password.html')

            

@app.route('/sign_in', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash("Username or email already exists!")
        else:
            new_user = User(username=username, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash("Account created successfully! Please log in.")
            return redirect(url_for('login'))
    return render_template('sign_in.html')

@app.route('/add_data', methods=['GET', 'POST'])
@login_required
def add_data():
    if request.method == 'POST':
        # Create new ESG record
        esg_record = ESGRecord(
            company=request.form['company'],
            location=request.form['location'],
            carbon_emissions=float(request.form['carbon_emissions']),
            diversity=float(request.form['diversity']),
            safety=float(request.form['safety']),
            compliance=float(request.form['compliance']),
            energy_efficiency=float(request.form['energy_efficiency']),
            user_id=current_user.id
        )
        
        # Calculate ESG score and rating
        esg_record.calculate_esg_score()
        
        # Add to database
        db.session.add(esg_record)
        db.session.commit()

        # Prepare data for blockchain
        blockchain_data = {
            'company': esg_record.company,
            'location': esg_record.location,
            'carbon_emissions': esg_record.carbon_emissions,
            'diversity': esg_record.diversity,
            'safety': esg_record.safety,
            'compliance': esg_record.compliance,
            'energy_efficiency': esg_record.energy_efficiency,
            'user': current_user.username,
            'esg_score': esg_record.esg_score,
            'esg_rating': esg_record.esg_rating
        }

        # Add block to blockchain with mining
        new_block = blockchain.add_new_block(blockchain_data)
        
        flash(f'Data added successfully! Mining time: {new_block["mining_time"]}')
        return redirect(url_for('index'))

    return render_template('add_data.html')

@app.route('/export_csv')
def export_csv():
    # Flatten blockchain data
    flattened_data = []
    for block in blockchain.chain:
        for record in block['data']:
            record_data = {
                'Block Index': block['index'],
                'Timestamp': block['timestamp'],
                'Company': record['company'],
                'Location': record['location'],
                'Carbon Emissions': record['carbon_emissions'],
                'Diversity': record['diversity'],
                'Safety': record['safety'],
                'Compliance': record['compliance'],
                'Energy Efficiency': record['energy_efficiency'],
                'ESG Score': record.get('esg_score', ''),
                'ESG Rating': record.get('esg_rating', ''),
                'Added By': record.get('user', ''),
                'Previous Hash': block['previous_hash']
            }
            flattened_data.append(record_data)
    
    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Create CSV in memory
    output = BytesIO()
    df.to_csv(output, index=False, encoding='utf-8')
    output.seek(0)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'esg_records_{timestamp}.csv'
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

@app.route('/export_excel')
def export_excel():
    # Flatten blockchain data
    flattened_data = []
    for block in blockchain.chain:
        for record in block['data']:
            record_data = {
                'Block Index': block['index'],
                'Timestamp': block['timestamp'],
                'Company': record['company'],
                'Location': record['location'],
                'Carbon Emissions': record['carbon_emissions'],
                'Diversity': record['diversity'],
                'Safety': record['safety'],
                'Compliance': record['compliance'],
                'Energy Efficiency': record['energy_efficiency'],
                'ESG Score': record.get('esg_score', ''),
                'ESG Rating': record.get('esg_rating', ''),
                'Added By': record.get('user', ''),
                'Previous Hash': block['previous_hash']
            }
            flattened_data.append(record_data)
    
    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Create Excel file in memory
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='ESG Records', index=False)
    
    # Auto-adjust columns width
    worksheet = writer.sheets['ESG Records']
    for idx, col in enumerate(df.columns):
        max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
        worksheet.set_column(idx, idx, max_length)
    
    writer.close()
    output.seek(0)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'esg_records_{timestamp}.xlsx'
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/api/verify/<int:block_index>')
def verify_block_api(block_index):
    """API endpoint for block verification"""
    block = None
    for b in blockchain.chain:
        if b['index'] == block_index:
            block = b
            break
    
    if not block:
        return jsonify({'error': 'Block not found'}), 404
        
    # Verify the block
    is_valid = True
    if block_index > 1:
        previous_block = blockchain.chain[block_index - 2]
        is_valid = (block['previous_hash'] == blockchain.hash(previous_block) and
                   blockchain.is_valid_proof(block['proof'], block['previous_hash']))
    
    return jsonify({
        'block': block,
        'is_valid': is_valid,
        'verification_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    })
    
COUNTRY_CODES = {
    # Cities/Locations to ISO country codes
    'dubai': 'AE',
    'abu dhabi': 'AE',
    'london': 'GB',
    'manchester': 'GB',
    'new york': 'US',
    'san francisco': 'US',
    'cupertino': 'US',
    'tokyo': 'JP',
    'osaka': 'JP',
    'seoul': 'KR',
    'beijing': 'CN',
    'shanghai': 'CN',
    'hong kong': 'HK',
    'singapore': 'SG',
    'sydney': 'AU',
    'melbourne': 'AU',
    'mumbai': 'IN',
    'bangalore': 'IN',
    'madrid': 'ES',
    'barcelona': 'ES',
    'paris': 'FR',
    'berlin': 'DE',
    'munich': 'DE',
    'amsterdam': 'NL',
    'toronto': 'CA',
    'vancouver': 'CA',
    'sao paulo': 'BR',
    'rio de janeiro': 'BR',
    
    # Countries (in case full country names are used)
    'united states': 'US',
    'united kingdom': 'GB',
    'united arab emirates': 'AE',
    'china': 'CN',
    'india': 'IN',
    'japan': 'JP',
    'south korea': 'KR',
    'australia': 'AU',
    'germany': 'DE',
    'france': 'FR',
    'spain': 'ES',
    'netherlands': 'NL',
    'canada': 'CA',
    'brazil': 'BR',
    'russia': 'RU',
    'singapore': 'SG'
}

def get_country_code(location):
    """
    Convert a location string to its ISO country code.
    
    Args:
        location (str): City or country name
        
    Returns:
        str: Two-letter ISO country code or 'UN' if not found
    """
    if not location:
        return 'UN'
    
    # Clean the location string
    location = location.lower().strip()
    
    # Direct lookup
    if location in COUNTRY_CODES:
        return COUNTRY_CODES[location]
    
    # Try to match partial names
    for loc, code in COUNTRY_CODES.items():
        if loc in location or location in loc:
            return code
            
    # Default return if no match found
    return 'UN'

# Example usage in your dashboard route:
def get_average_esg_scores_by_country(all_data):
    """
    Calculate average ESG scores by country from blockchain data.
    
    Args:
        all_data (list): List of ESG records from blockchain
        
    Returns:
        dict: Country codes mapped to average ESG scores
    """
    country_scores = {}
    country_counts = {}
    
    for record in all_data:
        country_code = get_country_code(record['location'])
        
        if country_code not in country_scores:
            country_scores[country_code] = 0
            country_counts[country_code] = 0
        
        country_scores[country_code] += float(record.get('esg_score', 0))
        country_counts[country_code] += 1
    
    # Calculate averages
    map_data = {
        code: scores / country_counts[code]
        for code, scores in country_scores.items()
    }
    
    return map_data

@app.route('/dashboard')
def dashboard():
    # Get all data from blockchain
    all_data = []
    for block in blockchain.chain:
        for record in block['data']:
            record['timestamp'] = block['timestamp']
            all_data.append(record)

    # Location-based aggregation
    location_data = {}
    for record in all_data:
        location = record['location']
        if location not in location_data:
            location_data[location] = {
                'esg_scores': [],
                'ratings': [],
                'carbon_emissions': [],
                'components': {
                    'diversity': [],
                    'safety': [],
                    'compliance': [],
                    'energy_efficiency': []
                }
            }
        
        location_data[location]['esg_scores'].append(float(record['esg_score']))
        location_data[location]['ratings'].append(record['esg_rating'])
        location_data[location]['carbon_emissions'].append(float(record['carbon_emissions']))
        location_data[location]['components']['diversity'].append(float(record['diversity']))
        location_data[location]['components']['safety'].append(float(record['safety']))
        location_data[location]['components']['compliance'].append(float(record['compliance']))
        location_data[location]['components']['energy_efficiency'].append(float(record['energy_efficiency']))

    # Company-based aggregation
    company_data = {}
    for record in all_data:
        company = record['company']
        if company not in company_data:
            company_data[company] = {
                'esg_scores': [],
                'ratings': [],
                'carbon_emissions': []
            }
        
        company_data[company]['esg_scores'].append(float(record['esg_score']))
        company_data[company]['ratings'].append(record['esg_rating'])
        company_data[company]['carbon_emissions'].append(float(record['carbon_emissions']))

    # Time-based aggregation
    timestamps = sorted(list(set(record['timestamp'] for record in all_data)))
    time_data = {
        'compliance': [],
        'esg_scores': []
    }
    
    for timestamp in timestamps:
        time_records = [r for r in all_data if r['timestamp'] == timestamp]
        time_data['compliance'].append(sum(float(r['compliance']) for r in time_records) / len(time_records))
        time_data['esg_scores'].append(sum(float(r['esg_score']) for r in time_records) / len(time_records))

    # Prepare chart data
    chart_data = {
        'locations': list(location_data.keys()),
        'companies': list(company_data.keys()),
        'timestamps': timestamps,
        'location_scores': [sum(data['esg_scores']) / len(data['esg_scores']) 
                          for data in location_data.values()],
        'company_scores': [sum(data['esg_scores']) / len(data['esg_scores']) 
                         for data in company_data.values()],
        'location_components': {
            loc: {
                comp: sum(data['components'][comp]) / len(data['components'][comp])
                for comp in data['components']
            }
            for loc, data in location_data.items()
        },
        'company_carbon': {
            comp: sum(data['carbon_emissions']) / len(data['carbon_emissions'])
            for comp, data in company_data.items()
        },
        'time_trends': {
            'compliance': time_data['compliance'],
            'esg_scores': time_data['esg_scores']
        },
        'raw_data': all_data  # Include raw data for detailed processing in frontend
    }

    
    print("Chart data preparation:")
    print("Number of locations:", len(chart_data['locations']))
    print("Number of companies:", len(chart_data['companies']))
    print("Number of timestamps:", len(chart_data['timestamps']))
    print("Location scores available:", len(chart_data['location_scores']))
    print("Company scores available:", len(chart_data['company_scores']))
    
    return render_template('dashboard.html', chart_data=chart_data)
    
from predictions import ESGPredictor

def calculate_esg_score(record):
    components = [
        float(record['carbon_emissions']),
        float(record['diversity']),
        float(record['safety']),
        float(record['compliance']),
        float(record['energy_efficiency'])
    ]
    return sum(components) / len(components)

@app.route('/predictions')
def predictions():
    # Get filter parameters
    filter_type = request.args.get('filter_type', 'all')
    filter_value = request.args.get('filter_value', None)

    # Get all data from blockchain
    all_data = []
    for block in blockchain.chain:
        for record in block['data']:
            record['timestamp'] = block['timestamp']
            # Calculate ESG score for each record
            record['esg_score'] = (
                float(record['carbon_emissions']) + 
                float(record['diversity']) + 
                float(record['safety']) + 
                float(record['compliance']) + 
                float(record['energy_efficiency'])
            ) / 5.0
            all_data.append(record)

    # Get unique companies and locations for filters
    companies = sorted(list(set(record['company'] for record in all_data)))
    locations = sorted(list(set(record['location'] for record in all_data)))
    company_locations = sorted(list(set(f"{record['company']} - {record['location']}" for record in all_data)))

    # Filter data based on selection
    filtered_data = []
    if filter_type == 'company' and filter_value:
        filtered_data = [record for record in all_data if record['company'] == filter_value]
    elif filter_type == 'location' and filter_value:
        filtered_data = [record for record in all_data if record['location'] == filter_value]
    elif filter_type == 'company_location' and filter_value:
        company, location = filter_value.split(' - ')
        filtered_data = [record for record in all_data if record['company'] == company and record['location'] == location]
    else:
        filtered_data = all_data

    if len(filtered_data) < 2:
        return render_template('predictions.html', 
                             error="Not enough data for predictions",
                             companies=companies,
                             locations=locations,
                             company_locations=company_locations,
                             filter_type=filter_type,
                             filter_value=filter_value)

    # Initialize predictor
    predictor = ESGPredictor(forecast_months=6)
    
    # Generate predictions for each metric including ESG score
    predictions = {}
    metrics = ['esg_score', 'carbon_emissions', 'diversity', 'safety', 'compliance', 'energy_efficiency']
    
    for metric in metrics:
        prediction = predictor.train_predict(filtered_data, metric)
        
        # Adjust trend calculation for carbon_emissions (lower is better)
        if metric == 'carbon_emissions':
            prediction['trend'] = 'Improving' if prediction['change_percent'] < 0 else 'Declining'
        else:
            prediction['trend'] = 'Improving' if prediction['change_percent'] > 0 else 'Declining'
            
        predictions[metric] = prediction

    return render_template('predictions.html',
                         predictions=predictions,
                         companies=companies,
                         locations=locations,
                         company_locations=company_locations,
                         filter_type=filter_type,
                         filter_value=filter_value)

@app.route('/download_template')
def download_template():
    """Download a template CSV file"""
    template_data = "Company,Location,Carbon Emissions,Diversity,Safety,Compliance,Energy Efficiency\n"
    template_data += "Example Corp,New York,75.5,85.2,92.1,78.9,88.3\n"
    
    return send_file(
        io.BytesIO(template_data.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='esg_data_template.csv'
    )

@app.route('/bulk_upload', methods=['POST'])
@login_required
def bulk_upload():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('add_data'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('add_data'))

    try:
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            flash('Invalid file format. Please upload CSV or Excel file.')
            return redirect(url_for('add_data'))

        # Validate columns
        required_columns = ['Company', 'Location', 'Carbon Emissions', 
                          'Diversity', 'Safety','Compliance', 'Energy Efficiency']
        
        # Convert column names to title case for comparison
        df.columns = [col.title() for col in df.columns]
        
        if not all(col in df.columns for col in required_columns):
            flash('Invalid file format. Please use the template provided.')
            return redirect(url_for('add_data'))

        # Process each row
        for _, row in df.iterrows():
            data = {
                'company': row['Company'],
                'location': row['Location'],
                'carbon_emissions': float(row['Carbon Emissions']),
                'diversity': float(row['Diversity']),
                'safety': float(row['Safety']),
                'compliance': float(row['Compliance']),
                'energy_efficiency': float(row['Energy Efficiency'])
            }

            # Create new ESG record
            esg_record = ESGRecord(
                user_id=current_user.id,
                **data
            )
            
            # Calculate ESG score and rating
            esg_record.calculate_esg_score()
            
            # Add to database
            db.session.add(esg_record)
            
            # Add to blockchain
            blockchain_data = {
                **data,
                'user': current_user.username,
                'esg_score': esg_record.esg_score,
                'esg_rating': esg_record.esg_rating
            }
            
            # Add block to blockchain
            blockchain.add_data(blockchain_data)

        # Commit database changes
        db.session.commit()
        
        # Create final block in blockchain
        previous_block = blockchain.get_previous_block()
        previous_hash = blockchain.hash(previous_block)
        proof, mining_time = blockchain.mine_block(previous_hash)
        blockchain.create_block(proof, previous_hash, mining_time)

        flash(f'Successfully added {len(df)} records')
        return redirect(url_for('index'))

    except Exception as e:
        flash(f'Error processing file: {str(e)}')
        return redirect(url_for('add_data'))

def reset_db():
    with app.app_context():
        db.drop_all()
        db.create_all()

# Call this once
reset_db()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
