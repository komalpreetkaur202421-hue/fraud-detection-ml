from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

try:
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    MODEL_LOADED = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    MODEL_LOADED = False

# ── Exact column order from X_train ──────────────────────────────────────────
ALL_FEATURES = [
    'age', 'policy_number', 'policy_bind_date', 'policy_state', 'policy_csl',
    'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'insured_zip',
    'insured_sex', 'insured_education_level', 'insured_occupation', 'insured_hobbies',
    'insured_relationship', 'capital-gains', 'capital-loss', 'incident_date',
    'incident_type', 'collision_type', 'incident_severity', 'authorities_contacted',
    'incident_state', 'incident_city', 'incident_location', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'property_damage', 'bodily_injuries', 'witnesses',
    'police_report_available', 'total_claim_amount', 'auto_make', 'auto_model',
    'auto_year', 'policy_annual_premium_log'
]

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Insurance Fraud Detector</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet"/>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Poppins', sans-serif;
    background: #f0f4ff;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 30px 16px;
  }
  .card {
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    width: 100%;
    max-width: 500px;
    overflow: hidden;
  }
  .card-header {
    background: linear-gradient(135deg, #1a3c8f, #2d5be3);
    color: white;
    padding: 24px 28px;
    text-align: center;
  }
  .card-header h1 { font-size: 1.4rem; font-weight: 700; }
  .card-header p  { font-size: 0.8rem; opacity: 0.8; margin-top: 4px; }
  .card-body { padding: 28px; }
  .field { margin-bottom: 16px; }
  label {
    display: block;
    font-size: 0.75rem;
    font-weight: 600;
    color: #444;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  input, select {
    width: 100%;
    padding: 10px 13px;
    border: 1.5px solid #dde3f0;
    border-radius: 8px;
    font-family: 'Poppins', sans-serif;
    font-size: 0.88rem;
    color: #222;
    background: #fafbff;
    outline: none;
    transition: border-color 0.2s;
  }
  input:focus, select:focus { border-color: #2d5be3; background: white; }
  .rupee-wrap { position: relative; }
  .rupee-wrap span {
    position: absolute; left: 12px; top: 50%;
    transform: translateY(-50%);
    font-size: 0.88rem; color: #666; font-weight: 600;
  }
  .rupee-wrap input { padding-left: 26px; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #2d5be3;
    margin: 20px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e8edff;
  }
  button {
    width: 100%;
    padding: 13px;
    background: linear-gradient(135deg, #1a3c8f, #2d5be3);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Poppins', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    margin-top: 8px;
    transition: opacity 0.2s;
  }
  button:hover { opacity: 0.9; }
  #result {
    display: none;
    margin-top: 20px;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  #result.fraud { background: #fff0f0; border: 2px solid #e53935; }
  #result.legit { background: #f0fff5; border: 2px solid #2e7d32; }
  #result .verdict { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
  #result.fraud .verdict { color: #e53935; }
  #result.legit .verdict { color: #2e7d32; }
  #result .sub { font-size: 0.82rem; color: #666; }
  #result .conf { margin-top: 12px; font-size: 0.85rem; color: #444; }
  .conf-bar-bg { height: 8px; background: #eee; border-radius: 4px; margin-top: 6px; overflow: hidden; }
  .conf-bar-fill { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
  .fraud .conf-bar-fill { background: #e53935; }
  .legit .conf-bar-fill { background: #2e7d32; }
  #error { color: #e53935; font-size: 0.82rem; margin-top: 10px; text-align: center; display: none; }
  .spinner { display: none; justify-content: center; margin-top: 12px; }
  .spinner div {
    width: 10px; height: 10px; margin: 0 4px;
    border-radius: 50%; background: #2d5be3;
    animation: bounce 0.6s infinite alternate;
  }
  .spinner div:nth-child(2) { animation-delay: 0.2s; }
  .spinner div:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { to { transform: translateY(-8px); opacity: 0.4; } }
</style>
</head>
<body>
<div class="card">
  <div class="card-header">
    <h1>&#128737; Insurance Fraud Detector</h1>
    <p>Enter claim details to check for fraud</p>
  </div>
  <div class="card-body">
    <form id="form">

      <!-- PERSON -->
      <div class="section-title">Person Details</div>
      <div class="row2">
        <div class="field">
          <label>Age</label>
          <input type="number" name="age" placeholder="e.g. 35" min="18" max="90" required/>
        </div>
        <div class="field">
          <label>Gender</label>
          <select name="insured_sex">
            <option value="MALE">Male</option>
            <option value="FEMALE">Female</option>
          </select>
        </div>
      </div>

      <!-- POLICY -->
      <div class="section-title">Policy Details</div>
      <div class="field">
        <label>Annual Premium (&#8377;)</label>
        <div class="rupee-wrap">
          <span>&#8377;</span>
          <input type="number" name="policy_annual_premium_inr" placeholder="e.g. 25000" min="500" required/>
        </div>
      </div>
      <div class="row2">
        <div class="field">
          <label>Policy Deductible (&#8377;)</label>
          <select name="policy_deductable">
            <option value="500">&#8377; 500</option>
            <option value="1000">&#8377; 1000</option>
            <option value="2000">&#8377; 2000</option>
          </select>
        </div>
        <div class="field">
          <label>Policy CSL</label>
          <select name="policy_csl">
            <option value="100/300">100/300</option>
            <option value="250/500">250/500</option>
            <option value="500/1000">500/1000</option>
          </select>
        </div>
      </div>

      <!-- INCIDENT -->
      <div class="section-title">Incident Details</div>
      <div class="row2">
        <div class="field">
          <label>Incident Date</label>
          <input type="date" name="incident_date" required/>
        </div>
        <div class="field">
          <label>Hour of Day (0-23)</label>
          <input type="number" name="incident_hour_of_the_day" min="0" max="23" placeholder="e.g. 14" required/>
        </div>
      </div>
      <div class="row2">
        <div class="field">
          <label>Incident Type</label>
          <select name="incident_type">
            <option value="Single Vehicle Collision">Single Vehicle</option>
            <option value="Multi-vehicle Collision">Multi Vehicle</option>
            <option value="Vehicle Theft">Theft</option>
            <option value="Parked Car">Parked Car</option>
          </select>
        </div>
        <div class="field">
          <label>Severity</label>
          <select name="incident_severity">
            <option value="Trivial Damage">Trivial</option>
            <option value="Minor Damage">Minor</option>
            <option value="Major Damage">Major</option>
            <option value="Total Loss">Total Loss</option>
          </select>
        </div>
      </div>
      <div class="row2">
        <div class="field">
          <label>Property Damage?</label>
          <select name="property_damage">
            <option value="YES">Yes</option>
            <option value="NO">No</option>
            <option value="?">Unknown</option>
          </select>
        </div>
        <div class="field">
          <label>Police Report?</label>
          <select name="police_report_available">
            <option value="YES">Yes</option>
            <option value="NO">No</option>
            <option value="?">Unknown</option>
          </select>
        </div>
      </div>
      <div class="row2">
        <div class="field">
          <label>Bodily Injuries</label>
          <select name="bodily_injuries">
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
          </select>
        </div>
        <div class="field">
          <label>Witnesses</label>
          <select name="witnesses">
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
          </select>
        </div>
      </div>
      <div class="field">
        <label>Authorities Contacted</label>
        <select name="authorities_contacted">
          <option value="Police">Police</option>
          <option value="Ambulance">Ambulance</option>
          <option value="Fire">Fire</option>
          <option value="None">None</option>
          <option value="Other">Other</option>
        </select>
      </div>

      <!-- CLAIM -->
      <div class="section-title">Claim Amount</div>
      <div class="field">
        <label>Total Claim Amount (&#8377;)</label>
        <div class="rupee-wrap">
          <span>&#8377;</span>
          <input type="number" name="total_claim_amount_inr" placeholder="e.g. 150000" min="0" required/>
        </div>
      </div>

      <button type="submit">&#128269; Check for Fraud</button>
      <div class="spinner" id="spinner"><div></div><div></div><div></div></div>
      <div id="error"></div>
    </form>

    <div id="result">
      <div class="verdict" id="verdict"></div>
      <div class="sub" id="sub"></div>
      <div class="conf">
        Confidence: <strong id="conf-val"></strong>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" id="conf-bar" style="width:0%"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  const INR_RATE = 83.5; // 1 USD = 83.5 INR

  document.getElementById('form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const spinner  = document.getElementById('spinner');
    const errorDiv = document.getElementById('error');
    const result   = document.getElementById('result');

    spinner.style.display = 'flex';
    errorDiv.style.display = 'none';
    result.style.display   = 'none';

    const data = Object.fromEntries(new FormData(e.target).entries());

    // Convert INR to USD (model trained on USD)
    const premiumUSD     = parseFloat(data.policy_annual_premium_inr) / INR_RATE;
    const claimUSD       = parseFloat(data.total_claim_amount_inr)    / INR_RATE;

    // Build payload with ALL required columns in correct order
    const payload = {
      age:                        data.age,
      policy_number:              '123456',
      policy_bind_date:           '2015-01-01',
      policy_state:               'OH',
      policy_csl:                 data.policy_csl,
      policy_deductable:          data.policy_deductable,
      policy_annual_premium:      premiumUSD.toFixed(2),
      umbrella_limit:             '0',
      insured_zip:                '432020',
      insured_sex:                data.insured_sex,
      insured_education_level:    'College',
      insured_occupation:         'other-service',
      insured_hobbies:            'reading',
      insured_relationship:       'self',
      'capital-gains':            '0',
      'capital-loss':             '0',
      incident_date:              data.incident_date,
      incident_type:              data.incident_type,
      collision_type:             'Front Collision',
      incident_severity:          data.incident_severity,
      authorities_contacted:      data.authorities_contacted,
      incident_state:             'OH',
      incident_city:              'Columbus',
      incident_location:          'Unknown',
      incident_hour_of_the_day:   data.incident_hour_of_the_day,
      number_of_vehicles_involved:'1',
      property_damage:            data.property_damage,
      bodily_injuries:            data.bodily_injuries,
      witnesses:                  data.witnesses,
      police_report_available:    data.police_report_available,
      total_claim_amount:         claimUSD.toFixed(2),
      auto_make:                  'Toyota',
      auto_model:                 'Camry',
      auto_year:                  '2015',
      policy_annual_premium_log:  Math.log(premiumUSD).toFixed(4)
    };

    try {
      const res  = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const json = await res.json();
      if (json.error) throw new Error(json.error);

      const isFraud = json.prediction === 'Fraud';
      const conf    = (json.confidence * 100).toFixed(1);

      result.className = isFraud ? 'fraud' : 'legit';
      document.getElementById('verdict').textContent = isFraud ? '&#9888; Fraud Detected' : '&#10003; Claim Looks Legitimate';
      document.getElementById('sub').textContent     = isFraud
        ? 'This claim shows signs of fraud. Further investigation recommended.'
        : 'No fraud indicators found. Claim appears genuine.';
      document.getElementById('conf-val').textContent = conf + '%';
      document.getElementById('conf-bar').style.width = conf + '%';
      result.style.display = 'block';
      result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch(err) {
      errorDiv.textContent   = 'Error: ' + (err.message || 'Something went wrong.');
      errorDiv.style.display = 'block';
    } finally {
      spinner.style.display = 'none';
    }
  });
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded. Place .pkl files in same folder as app.py'}), 500

    try:
        data = request.get_json()
        row  = {}

        for col in ALL_FEATURES:
            val = str(data.get(col, ''))
            le  = label_encoders.get(col)
            if le is not None:
                # Categorical — encode with saved LabelEncoder
                row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
            else:
                # Numeric
                try:
                    row[col] = float(val)
                except (ValueError, TypeError):
                    row[col] = 0.0

        input_df        = pd.DataFrame([row], columns=ALL_FEATURES)
        input_scaled    = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=ALL_FEATURES)

        pred       = model.predict(input_scaled_df)[0]
        confidence = float(max(model.predict_proba(input_scaled_df)[0])) \
                     if hasattr(model, 'predict_proba') else 1.0

        fraud_le = label_encoders.get('fraud_reported')
        if fraud_le:
            label           = fraud_le.inverse_transform([pred])[0]
            prediction_text = 'Fraud' if label == 'Y' else 'No Fraud'
        else:
            prediction_text = 'Fraud' if pred == 1 else 'No Fraud'

        return jsonify({'prediction': prediction_text, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("=" * 50)
    print("  Insurance Fraud Detector")
    print("  Open http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
