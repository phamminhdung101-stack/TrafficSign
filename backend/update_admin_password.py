import sys
import os

# Add backend dir to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pwdlib import PasswordHash
from sqlalchemy import create_engine, text

engine = create_engine('mysql+pymysql://root:@localhost:3306/traffic_sign_db')
password_hash = PasswordHash.recommended()
hashed = password_hash.hash('123456')

with engine.begin() as conn:
    conn.execute(text('UPDATE users SET password_hash = :hash WHERE username = :user'), {'hash': hashed, 'user': 'admin'})

print("Updated password for admin")
