import csv

from werkzeug.security import generate_password_hash

USER_CSV_FILE = 'users.csv'

def save_user_to_csv(email, password):
    """将用户保存到 CSV 文件"""
    with open(USER_CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([email, generate_password_hash(password)])


def get_user_from_csv(email):
    """从 CSV 文件中读取用户信息"""
    with open(USER_CSV_FILE, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['email'] == email:
                return row
    return None