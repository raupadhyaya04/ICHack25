// seedTransactionWithCSV.js
const fs = require('fs');
const csv = require('csv-parser');
const mongoose = require('mongoose');
const { faker } = require('@faker-js/faker');
const Transaction = require('./models/Transaction');

// Connection string (update with your credentials and database name)
const CONNECTION_STRING = 'mongodb+srv://bartekwatrobinski:v3LC4LfQnWwR1M3l@cluster0.upddr.mongodb.net/hackathonDB?retryWrites=true&w=majority';

// Cost per resource unit
const COST_PER_UNIT = 7;

// Arrays to hold CSV data
const foodBanks = [];
const markets = [];

// Function to load food banks from CSV
function loadFoodBanksFromCSV() {
  return new Promise((resolve, reject) => {
    fs.createReadStream('data_processing/demand_timeseries.csv') // Adjust path if needed
      .pipe(csv())
      .on('data', (row) => {
        // Expecting columns like: name, Day1, Day2, ... DayN
        const dailyDemands = {};
        for (const key in row) {
          if (key.startsWith('Day')) {
            // Parse the value; if invalid, default to 0.
            const value = parseFloat(row[key]);
            dailyDemands[key] = isNaN(value) ? 0 : value;
          }
        }
        foodBanks.push({
          name: row.name,
          dailyDemands // an object mapping day labels to demand values
        });
      })
      .on('end', () => {
        console.log('Food bank CSV processed. Total food banks:', foodBanks.length);
        resolve();
      })
      .on('error', (err) => reject(err));
  });
}

// Function to load markets from CSV
function loadMarketsFromCSV() {
  return new Promise((resolve, reject) => {
    fs.createReadStream('data_processing/clearer_ish_data.csv') // Adjust path if needed
      .pipe(csv())
      .on('data', (row) => {
        // Convert numeric fields; if invalid, default to 0.
        row.demand = isNaN(parseFloat(row.demand)) ? 0 : parseFloat(row.demand);
        row.supply = isNaN(parseFloat(row.supply)) ? 0 : parseFloat(row.supply);
        row.waste = isNaN(parseFloat(row.waste)) ? 0 : parseFloat(row.waste);
        // Generate a composite unique ID (optional)
        row.id = `${row.name}-${row.lat}-${row.lng}`;
        markets.push(row);
      })
      .on('end', () => {
        console.log('Market CSV processed. Total markets:', markets.length);
        resolve();
      })
      .on('error', (err) => reject(err));
  });
}

// Connect to MongoDB and run the seed process
mongoose.connect(CONNECTION_STRING, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
  .then(async () => {
    console.log('Connected to MongoDB Atlas!');
    await Transaction.deleteMany({});
    console.log('Cleared existing transactions.');
    await loadMarketsFromCSV();
    await loadFoodBanksFromCSV();
    await generateTransactions();
  })
  .catch(err => console.error('Error connecting to MongoDB Atlas:', err))
  .finally(() => {
    mongoose.connection.close();
    console.log('Connection closed.');
  });

// Transaction generation function using CSV data and enforcing constraints
async function generateTransactions() {
  const maxTransactions = 100;
  let transactionCount = 0;
  
  // Ensure both markets and foodBanks arrays are loaded.
  if (markets.length === 0 || foodBanks.length === 0) {
    console.error('Market or food bank data not loaded.');
    return;
  }
  
  // Track each food bank's daily totals, e.g. { 'FoodBank A': { 'Day 1': 0, 'Day 2': 0, ... } }
  const foodBankDailyTotals = {};
  
  // Helper function: pick a random element from an array.
  const getRandomElement = (arr) => arr[Math.floor(Math.random() * arr.length)];
  
  while (transactionCount < maxTransactions) {
    const market = getRandomElement(markets);
    const foodBankData = getRandomElement(foodBanks);
    
    // Pick a random day label (assumed keys like 'Day 1', 'Day 2', etc.)
    const dayKeys = Object.keys(foodBankData.dailyDemands);
    const dayLabel = getRandomElement(dayKeys);
    
    // Get the food bank's demand for that day.
    const dailyDemand = foodBankData.dailyDemands[dayLabel];
    
    // Initialize tracking for this food bank if needed.
    if (!foodBankDailyTotals[foodBankData.name]) {
      foodBankDailyTotals[foodBankData.name] = {};
    }
    if (!foodBankDailyTotals[foodBankData.name][dayLabel]) {
      foodBankDailyTotals[foodBankData.name][dayLabel] = 0;
    }
    
    // Calculate how many units remain for that food bank on that day.
    const remainingFoodBankUnits = dailyDemand - foodBankDailyTotals[foodBankData.name][dayLabel];
    if (remainingFoodBankUnits <= 0) continue;
    
    // Limit transaction units by the market's waste supply.
    const maxUnitsForMarket = Math.min(10, Math.floor(market.waste));
    
    // Compute the possible maximum units allowed by both constraints.
    const possibleMaxUnits = Math.min(maxUnitsForMarket, remainingFoodBankUnits);
    
    // If possibleMaxUnits is not a finite number or is less than 1, skip this iteration.
    if (!isFinite(possibleMaxUnits) || possibleMaxUnits < 1) continue;
    
    // Generate a random number of units between 1 and possibleMaxUnits.
    const units = faker.number.int({ min: 1, max: possibleMaxUnits });
    // Ensure units is a valid number.
    if (!isFinite(units) || units < 1) continue;
    
    const amount = units * COST_PER_UNIT;
    
    // Create a transaction date corresponding to the chosen day.
    let transactionDate = new Date('2025-02-01');
    // Extract day number from dayLabel, e.g., "Day 1" -> 1
    const dayNumber = parseInt(dayLabel.split(' ')[1]);
    transactionDate.setDate(transactionDate.getDate() + dayNumber - 1);
    
    // Create and save the transaction.
    const transaction = new Transaction({
      store: market.name,
      foodBank: foodBankData.name,
      amount,
      resourceUnits: units,
      date: transactionDate
    });
    
    await transaction.save();
    transactionCount++;
    foodBankDailyTotals[foodBankData.name][dayLabel] += units;
    
    console.log(`Transaction ${transactionCount}: ${units} units from ${market.name} to ${foodBankData.name} on ${dayLabel}.`);
  }
  console.log(`Generated ${transactionCount} transactions.`);
}
