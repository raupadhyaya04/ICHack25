// models/Transaction.js
const mongoose = require('mongoose');

const transactionSchema = new mongoose.Schema({
  store: { type: String, required: true },
  foodBank: { type: String, required: true },
  amount: { type: Number, required: true },
  resourceUnits: { type: Number, required: true },
  date: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Transaction', transactionSchema);
