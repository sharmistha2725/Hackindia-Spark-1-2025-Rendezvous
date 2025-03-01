from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
from severity_model import severity_model
import razorpay
import os
from server.blockchain import blockchain

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_fund.db'
db = SQLAlchemy(app)

# Initialize Razorpay client
razorpay_client = razorpay.Client(
    auth=(os.environ.get('RAZORPAY_KEY_ID'), 
          os.environ.get('RAZORPAY_KEY_SECRET'))
)

# Database Models
class Campaign(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    amount_needed = db.Column(db.Float, nullable=False)
    amount_raised = db.Column(db.Float, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    severity_score = db.Column(db.Integer)
    ml_confidence = db.Column(db.Float)
    severity_class = db.Column(db.String(20))
    donations = db.relationship('Donation', backref='campaign', lazy=True)

class Donation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float, nullable=False)
    donor_name = db.Column(db.String(80))
    payment_id = db.Column(db.String(100))  # Razorpay payment ID
    campaign_id = db.Column(db.Integer, db.ForeignKey('campaign.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    block_hash = db.Column(db.String(64))  # Hash of the blockchain block

@app.route('/')
def home():
    campaigns = Campaign.query.order_by(Campaign.severity_score.desc()).all()
    return render_template('home.html', campaigns=campaigns)

@app.route('/create_campaign', methods=['GET', 'POST'])
def create_campaign():
    if request.method == 'POST':
        condition = request.form['condition']
        symptoms = request.form['symptoms']
        diagnosis = request.form['diagnosis']

        # Calculate severity using condition, symptoms, and diagnosis
        severity_result = severity_model.predict_severity(
            condition=condition,
            symptoms=symptoms,
            diagnosis=diagnosis
        )

        campaign = Campaign(
            title=condition,  # Use condition as the title
            description=f"Condition: {condition}\nSymptoms: {symptoms}\nDiagnosis: {diagnosis}",
            amount_needed=float(request.form['amount_needed']),
            severity_score=severity_result['severityScore'],
            ml_confidence=severity_result['confidence'],
            severity_class=severity_result['classification']
        )

        db.session.add(campaign)
        db.session.commit()
        flash('Campaign created successfully!')
        return redirect(url_for('home'))

    return render_template('create_campaign.html')

@app.route('/donate/<int:campaign_id>', methods=['GET'])
def donate(campaign_id):
    campaign = Campaign.query.get_or_404(campaign_id)
    return render_template('donate.html', campaign=campaign, 
                         key_id=os.environ.get('RAZORPAY_KEY_ID'))

@app.route('/create_order/<int:campaign_id>', methods=['POST'])
def create_order(campaign_id):
    campaign = Campaign.query.get_or_404(campaign_id)
    amount = int(float(request.form['amount']) * 100)  # Convert to paise

    order_data = {
        'amount': amount,
        'currency': 'INR',
        'receipt': f'order_rcptid_{campaign_id}_{datetime.now().timestamp()}'
    }

    order = razorpay_client.order.create(data=order_data)
    return jsonify(order)

@app.route('/payment_callback', methods=['POST'])
def payment_callback():
    payment_id = request.form.get('razorpay_payment_id')
    order_id = request.form.get('razorpay_order_id')
    signature = request.form.get('razorpay_signature')
    campaign_id = int(request.form.get('campaign_id'))
    amount = float(request.form.get('amount'))
    donor_name = request.form.get('donor_name')

    try:
        # Verify Razorpay payment
        razorpay_client.utility.verify_payment_signature({
            'razorpay_payment_id': payment_id,
            'razorpay_order_id': order_id,
            'razorpay_signature': signature
        })

        # Record transaction in blockchain
        transaction_data = {
            "payment_id": payment_id,
            "campaign_id": campaign_id,
            "amount": amount,
            "donor_name": donor_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        block = blockchain.create_block(transaction_data)

        # Create donation record
        campaign = Campaign.query.get(campaign_id)
        donation = Donation(
            amount=amount,
            donor_name=donor_name,
            payment_id=payment_id,
            campaign_id=campaign_id,
            block_hash=block.hash  # Store block hash for reference
        )
        campaign.amount_raised += amount

        db.session.add(donation)
        db.session.commit()

        flash('Thank you for your donation!')
        return redirect(url_for('home'))

    except Exception as e:
        flash('Payment verification failed')
        return redirect(url_for('donate', campaign_id=campaign_id))

@app.route('/api/transactions')
def get_transactions():
    """API endpoint to view all blockchain transactions"""
    return jsonify(blockchain.get_all_transactions())

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5001, debug=True)