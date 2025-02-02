// pages/Suppliers.js
import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import '../app.css';

function Suppliers() {
    // Sample state for form inputs
    const [productData, setProductData] = useState({
      name: '',
      quantity: '',
      date: '',
    });
  
    // Sample statistics data
    const stats = {
      peopleServed: 2500,
      creditsGenerated: 1200,
      wasteSaved: "5000kg",
      co2Savings: "3.5 tonnes"
    };
  
    // Sample distributors data
    const distributors = [
      { name: "Food Bank A", capacity: "1000kg", window: "9AM - 5PM" },
      { name: "Community Center B", capacity: "500kg", window: "10AM - 3PM" },
      { name: "Charity C", capacity: "750kg", window: "8AM - 4PM" },
    ];

    const creditStats = [
      { Fact: "Your Waste Credit balance is: 20"},
      { Fact: "Your Food Credits help millions of hungry people eat"}
    ]
  
    return (
      <div>
        <div className="header">
          <h1 className = 'title'>ReSource</h1>
          <Navbar />
        </div>
        `<div className="distributors-page">
          <div className="dashboard-container">
            {/* Left Column */}
            <div className="left-column">
              {/* Product Updates Section */}
              <div className="product-updates">
                <h2>Update Products</h2>
                <form className="update-form">
                  <input 
                    type="text" 
                    placeholder="Product Name"
                    value={productData.name}
                    onChange={(e) => setProductData({...productData, name: e.target.value})}
                  />
                  <input 
                    type="number" 
                    placeholder="Quantity"
                    value={productData.quantity}
                    onChange={(e) => setProductData({...productData, quantity: e.target.value})}
                  />
                  <input 
                    type="date" 
                    value={productData.date}
                    onChange={(e) => setProductData({...productData, date: e.target.value})}
                  />
                  <button type="submit">Update</button>
                </form>
              </div>
    
              {/* Statistics Section */}
              <div className="statistics">
                <h2>Impact Statistics</h2>
                <div className="stats-grid">
                  <div className="stat-box">
                    <h3>People Fed</h3>
                    <p>{stats.peopleServed}</p>
                    <small>*Based on 2kg per person estimate</small>
                  </div>
                  <div className="stat-box">
                    <h3>Credits Generated</h3>
                    <p>{stats.creditsGenerated}</p>
                  </div>
                  <div className="stat-box">
                    <h3>Waste Saved</h3>
                    <p>{stats.wasteSaved}</p>
                  </div>
                  <div className="stat-box">
                    <h3>CO2 Savings</h3>
                    <p>{stats.co2Savings}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Upper Right Column - Credit Stats Table */}
            <div className="right-column">
                <table className="credStats-table">
                  <thead>
                    <tr>
                      <th>Food Waste Credit Metrics</th>
                    </tr>
                  </thead>
                  <tbody>
                    {creditStats.map((creditStat, index) => (
                      <tr key={index}>
                        <div className="stat-box">
                        <td>{creditStat.Fact}</td>
                        </div>
                      </tr>
                    ))}
                  </tbody>
                </table>
            {/* Lower Right Column - Distributors Table */}
              <table className="distributors-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Capacity</th>
                    <th>Window of Time</th>
                  </tr>
                </thead>
                <tbody>
                  {distributors.map((distributor, index) => (
                    <tr key={index}>
                      <td>{distributor.name}</td>
                      <td>{distributor.capacity}</td>
                      <td>{distributor.window}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    );
}

export default Suppliers;

