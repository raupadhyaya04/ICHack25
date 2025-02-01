// src/pages/Distributors.js
import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import '../app.css';

// You might want to move these to a separate data file
const mockShopData = [
  { id: 1, name: "Shop A", collectionsCount: 15, totalAmount: 750 },
  { id: 2, name: "Shop B", collectionsCount: 8, totalAmount: 400 },
  // Add more mock data as needed
];

const mockAppointments = [
  { id: 1, date: "2024-01-20", time: "10:00", location: "Shop A", status: "available" },
  { id: 2, date: "2024-01-21", time: "14:30", location: "Shop B", status: "booked" },
  // Add more mock data as needed
];

const BreakdownTable = () => {
  return (
    <div className="breakdown-table">
      <h2>Collections Breakdown</h2>
      <table>
        <thead>
          <tr>
            <th>Shop Name</th>
            <th>Collections</th>
            <th>Total Amount</th>
          </tr>
        </thead>
        <tbody>
          {mockShopData.map(shop => (
            <tr key={shop.id}>
              <td>{shop.name}</td>
              <td>{shop.collectionsCount}</td>
              <td>${shop.totalAmount}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const AppointmentsList = () => {
  const handleBooking = (appointmentId) => {
    // Implement booking logic here
    console.log(`Booking appointment ${appointmentId}`);
  };

  return (
    <div className="appointments-list">
      <h2>Available Appointments</h2>
      <div className="appointments-grid">
        {mockAppointments.map(appointment => (
          <div key={appointment.id} className="appointment-card">
            <div className="appointment-details">
              <p>Date: {appointment.date}</p>
              <p>Time: {appointment.time}</p>
              <p>Location: {appointment.location}</p>
            </div>
            <button
              onClick={() => handleBooking(appointment.id)}
              disabled={appointment.status === 'booked'}
              className={`booking-button ${appointment.status === 'booked' ? 'booked' : ''}`}
            >
              {appointment.status === 'booked' ? 'Booked' : 'Book Now'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default function Distributors() {
  const [credits, setCredits] = useState(0);

  return (
    <div>
      <div className="header">
            <h1 className = 'title'>ReSource</h1>
            <Navbar />
          </div>
      <div className="distributors-page">
        <div className="content-container">
          <BreakdownTable />
          <AppointmentsList />
        </div>
      </div>
    </div>
  );
}