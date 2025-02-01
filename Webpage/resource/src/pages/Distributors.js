// src/pages/Distributors.js
import React, { useState } from 'react';
import SupplyForm from '../components/SupplyForm';
import DistributorsDashboard from '../components/DistributorsDashboard';
import Statistics from '../components/Statistics';

export default function Distributors() {
  const [credits, setCredits] = useState(0);

  return (
    <div className="suppliers-page">
      <div className="credits-counter">Credits: {credits}</div>
      <SupplyForm />
      <DistributorsDashboard />
      <Statistics />
    </div>
  );
}