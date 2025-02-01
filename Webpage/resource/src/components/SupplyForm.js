// components/SupplyForm.js
import React, { useState } from 'react';
import './SupplyForm.css';

function SupplyForm() {
  const [supplyData, setSupplyData] = useState({
    quantity: '',
    type: '',
    expiryDate: '',
    description: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Supply Data:', supplyData);
    // Handle form submission
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setSupplyData(prevState => ({
      ...prevState,
      [name]: value
    }));
  };

  return (
    <form className="supply-form" onSubmit={handleSubmit}>
      <h2>Supply Form</h2>
      <div className="form-group">
        <label htmlFor="quantity">Quantity:</label>
        <input
          type="number"
          id="quantity"
          name="quantity"
          value={supplyData.quantity}
          onChange={handleChange}
          required
        />
      </div>
      <div className="form-group">
        <label htmlFor="type">Type:</label>
        <select
          id="type"
          name="type"
          value={supplyData.type}
          onChange={handleChange}
          required
        >
          <option value="">Select Type</option>
          <option value="food">Food</option>
          <option value="clothing">Clothing</option>
          <option value="medical">Medical Supplies</option>
        </select>
      </div>
      <div className="form-group">
        <label htmlFor="expiryDate">Expiry Date:</label>
        <input
          type="date"
          id="expiryDate"
          name="expiryDate"
          value={supplyData.expiryDate}
          onChange={handleChange}
          required
        />
      </div>
      <div className="form-group">
        <label htmlFor="description">Description:</label>
        <textarea
          id="description"
          name="description"
          value={supplyData.description}
          onChange={handleChange}
          rows="4"
        />
      </div>
      <button type="submit">Submit Supply</button>
    </form>
  );
}

export default SupplyForm;