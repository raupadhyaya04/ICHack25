// seed.js

const mongoose = require('mongoose');
const { faker } = require('@faker-js/faker');
const Transaction = require('./models/Transaction');


const CONNECTION_STRING = 'mongodb+srv://bartekwatrobinski:v3LC4LfQnWwR1M3l@cluster0.upddr.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';

mongoose.connect(CONNECTION_STRING, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
  .then(() => {
    console.log('Connected to MongoDB Atlas!');
    seedTransactions();
  })
  .catch(err => console.error('Error connecting to MongoDB Atlas:', err));

const seedTransactions = async () => {
  try {
    // Optionally clear existing transactions
    await Transaction.deleteMany({});
    console.log('Cleared existing transactions.');

    // Create, for example, 100 fake transactions
    for (let i = 0; i < 100; i++) {
      const amount = parseFloat(faker.finance.amount(100, 1000, 2)); // random amount between 100 and 1000
      const newTransaction = new Transaction({
        supplier: faker.company.name(),
        distributor: faker.company.name(),
        amount: amount,
        // Calculate resource units using a cost per unit of 7 (as in your statistics logic)
        resourceUnits: Math.floor(amount / 7),
        date: faker.date.past(1) // random date within the past year
      });
      await newTransaction.save();
    }
    console.log('Fake transactions seeded successfully.');
  } catch (error) {
    console.error('Error seeding transactions:', error);
  } finally {
    mongoose.connection.close();
    console.log('Connection closed.');
  }
};
