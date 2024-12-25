from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class ESGRecord(db.Model):
    __tablename__ = 'esg_record'
    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(150), nullable=False)
    location = db.Column(db.String(150), nullable=False)
    carbon_emissions = db.Column(db.Float, nullable=False)
    diversity = db.Column(db.Float, nullable=False)
    safety = db.Column(db.Float, nullable=False)
    compliance = db.Column(db.Float, nullable=False)
    energy_efficiency = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    esg_score = db.Column(db.Float)
    esg_rating = db.Column(db.String(3))

    @property
    def formatted_timestamp(self):
        """Return timestamp in a readable format"""
        return self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')

    def calculate_esg_score(self):
        weights = {
            'carbon_emissions': 0.25,
            'diversity': 0.2,
            'safety': 0.2,
            'compliance': 0.15,
            'energy_efficiency': 0.2
        }
        
        carbon_score = 100 - min(self.carbon_emissions, 100)
        
        score = (
            weights['carbon_emissions'] * carbon_score +
            weights['diversity'] * self.diversity +
            weights['safety'] * self.safety +
            weights['compliance'] * self.compliance +
            weights['energy_efficiency'] * self.energy_efficiency
        )
        
        self.esg_score = round(score, 2)
        
        if score >= 90:
            self.esg_rating = 'AAA'
        elif score >= 80:
            self.esg_rating = 'AA'
        elif score >= 70:
            self.esg_rating = 'A'
        elif score >= 60:
            self.esg_rating = 'BBB'
        elif score >= 50:
            self.esg_rating = 'BB'
        elif score >= 40:
            self.esg_rating = 'B'
        else:
            self.esg_rating = 'CCC'

def init_db(app):
    with app.app_context():
        db.init_app(app)
        db.create_all()