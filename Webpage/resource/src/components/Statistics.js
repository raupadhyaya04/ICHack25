// src/components/Statistics.js
import React from 'react';

const COST_PER_UNIT = 7; // sample data
const WASTE_SAVED_PER_UNIT = 0.5; // sample
const CO2_SAVED_PER_UNIT = 1.2; // sample


const calculateResourceUnits = (totalAmountSpent) => {
  return Math.floor(totalAmountSpent / COST_PER_UNIT);
};

const calculateWasteSaved = (totalAmountSpent) => {
  const units = calculateResourceUnits(totalAmountSpent);
  return units * WASTE_SAVED_PER_UNIT;
};
const calculateCO2Savings = (totalAmountSpent) => {
  const units = calculateResourceUnits(totalAmountSpent);
  return units * CO2_SAVED_PER_UNIT;
};

const Statistics = ({ totalAmountSpent }) => {
  return (
    <div>
      <h2>Statistics</h2>
      <p>Waste Saved: {calculateWasteSaved(totalAmountSpent)} kg</p>
      <p>CO₂ Savings: {calculateCO2Savings(totalAmountSpent)} kg</p>
    </div>
  );
};

export default Statistics;