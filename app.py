import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
from extensions import db

load_dotenv()
# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass


# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "e4f8b7a2d9c349c3a5e71f6b20a3d1e8f57c3a99b0d4e2f8c1a7d9b6e4c3f0a2")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "postgresql://neondb_owner:npg_wGYhLirK26eO@ep-billowing-surf-a5v0vo7k-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models to ensure tables are created
    import models
    db.create_all()

# Import routes
import routes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
