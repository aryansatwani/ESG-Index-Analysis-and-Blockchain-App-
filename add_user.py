from models_esg import db, User
from blockchain_app import app

with app.app_context():
    user = User(username='user1', email='user1@example.com', password='password1')
    db.session.add(user)
    db.session.commit()
    print("User added successfully!")
