import os

class Config:
    # Secret key for forms and Flask-Login
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'

    # Database configuration (SQLite for simplicity)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///esg_app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

# Optional environment configurations
class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
