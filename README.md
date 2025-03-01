# Medical Crowdfunding Platform - Local Setup Guide

## System Requirements
Your system (12th Gen Intel, 16GB RAM, 64-bit OS) meets all the requirements for running this application. Here are the specific requirements:

- CPU: Intel/AMD x64 processor (your 12th Gen Intel is perfect)
- RAM: Minimum 8GB (your 16GB is ideal)
- OS: 64-bit Windows/Linux/MacOS (your 64-bit OS is compatible)
- Storage: At least 2GB free space for the application and dependencies
- Internet: Stable connection for Razorpay integration and blockchain features

## Installation Prerequisites

1. **Node.js (version 20)**:
   - Download from [nodejs.org](https://nodejs.org)
   - Verify installation: `node --version`

2. **PostgreSQL**:
   - Download from [postgresql.org](https://www.postgresql.org/download/)
   - During installation:
     - Use default port 5432
     - Set a secure password
     - Allow connections from localhost

3. **Python 3.11**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Add Python to PATH during installation
   - Verify: `python --version`

4. **Git**:
   - Download from [git-scm.com](https://git-scm.com/downloads)
   - Verify: `git --version`

## Project Setup

1. **Clone and Setup**:
```bash
git clone <repository-url>
cd medical-crowdfunding
npm install
```

2. **Environment Setup**:
Create a `.env` file in the project root:
```env
# Database Configuration
DATABASE_URL=postgres://username:password@localhost:5432/medical_fund_db

# Session Secret (Generate a random string)
SESSION_SECRET=your_random_secret_here

# Razorpay Configuration
RAZORPAY_KEY_ID=your_razorpay_key_id
RAZORPAY_KEY_SECRET=your_razorpay_secret_key

# Blockchain Configuration
VITE_CONTRACT_ADDRESS=your_deployed_contract_address
```

3. **Install Python Dependencies**:
```bash
pip install -r requirements.txt
```

## Key Components Documentation

### 1. ML Model Components
- **Location**: `server/services/ml-severity-model.ts`
- **Purpose**: Processes medical condition data to determine severity scores
- **Dependencies**: TensorFlow.js, Universal Sentence Encoder
- **Key Features**:
  - Text preprocessing
  - BioBERT model integration
  - Severity classification (mild/severe/critical)
  - Confidence scoring

### 2. Blockchain Integration
- **Location**: `server/blockchain.py`
- **Purpose**: Records donation transactions in blockchain
- **Features**:
  - Transaction verification
  - Block creation and chaining
  - Hash generation
  - Transaction history

### 3. Payment Integration
- **Location**: `templates/donate.html`
- **Purpose**: Handles Razorpay payment processing
- **Features**:
  - Payment gateway integration
  - Transaction verification
  - Receipt generation

### 4. Frontend Components
- **Location**: `templates/`
- **Key Files**:
  - `home.html`: Campaign listing
  - `create_campaign.html`: Campaign creation form
  - `donate.html`: Donation interface

### 5. Backend API
- **Location**: `app.py`
- **Purpose**: Main application server
- **Endpoints**:
  - `/`: Home page
  - `/create_campaign`: Campaign creation
  - `/donate/<campaign_id>`: Donation processing
  - `/api/transactions`: Blockchain records

## Running the Application

1. **Start PostgreSQL**:
   - Ensure PostgreSQL service is running
   - Create database: `createdb medical_fund_db`

2. **Initialize Database**:
```bash
npm run db:push
```

3. **Start Development Server**:
```bash
npm run dev
```

4. **Access Application**:
   - Open browser: `http://localhost:5000`
   - First-time setup:
     1. Create a test campaign
     2. Test donation flow with Razorpay test credentials

## Troubleshooting

1. **Database Connection Issues**:
   - Verify PostgreSQL is running: `pg_isready`
   - Check DATABASE_URL is correct
   - Ensure database exists

2. **ML Model Issues**:
   - Check TensorFlow.js installation
   - Verify model files in `models/` directory
   - Monitor console for prediction errors

3. **Payment Integration**:
   - Verify Razorpay credentials
   - Check network connectivity
   - Monitor payment callbacks

4. **Port Conflicts**:
   - If port 5000 is in use:
     1. Kill existing process: `lsof -i :5000`
     2. Or modify port in `server/index.ts`

## Project Structure
```
├── client/              # Frontend React application
│   └── src/
│       ├── components/  # UI components
│       ├── hooks/      # Custom React hooks
│       ├── lib/        # Utilities
│       └── pages/      # Page components
├── server/             # Backend services
│   ├── services/       # ML and blockchain services
│   └── routes.ts      # API routes
├── templates/          # HTML templates
└── shared/            # Shared code
    └── schema.ts      # Database schema
```

## Development Guidelines

1. **Code Style**:
   - Use TypeScript for type safety
   - Follow ESLint configuration
   - Run `npm run lint` before commits

2. **Testing**:
   - Run unit tests: `npm test`
   - Test ML model: `npm run test:ml`
   - Test blockchain: `npm run test:blockchain`

3. **Security**:
   - Never commit .env files
   - Use environment variables
   - Validate all user inputs

## Support and Resources
- Report issues on the repository
- Check documentation in `/docs`
- Contact support for Razorpay integration help

Feel free to reach out if you need any clarification or assistance with the setup!