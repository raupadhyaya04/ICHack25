// src/pages/Home.js
import React from 'react';
import Navbar from '../components/Navbar';
import homeImage from '../assets/nodes_photo.png';
import '../app.css';

export default function Home() {
  return (
    <div>
      <div className="header">
        <h1 className="title">ReSource</h1>
        <Navbar />
      </div>
      <div className="home-container">
        <div className="home-image-section">
          <img 
            src={homeImage} 
            alt="Food bank volunteers sorting donations"
            className="home-image"
          />
        </div>
        <div className="home-text-section">
          <h2>Connecting Surplus to Need</h2>
          <p>
            ReSource is an innovative platform that bridges the gap between supermarkets 
            with surplus food and organizations that help people in need.
          </p>
          <p>
            Our digital clearinghouse simplifies the process of food donation, making it 
            easier for supermarkets to reduce waste while helping food banks and charities 
            access fresh, quality food for their communities.
          </p>
          <p>
            By connecting suppliers directly with local food banks and charitable organisations, 
            we're creating a more efficient and sustainable way to fight food insecurity while 
            reducing food waste.
          </p>
          <div className="cta-section">
          <a href="/suppliers">
            <button className="cta-button">Join as a Supplier</button>
          </a>
          <a href="distributors">
            <button className="cta-button">Register as a Food Bank</button>
          </a>
          </div>
        </div>
      </div>
    </div>
  );
}