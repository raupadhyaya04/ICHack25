// components/Navigation.js
import React from 'react';
import { Link } from 'react-router-dom';

function Navbar() {
  return (
    <nav className="nav-bar">
      <ul>
        <li><Link to="/">Home</Link></li>
        <li><Link to="/suppliers">Suppliers</Link></li>
        <li><Link to="/distributors">Distributors</Link></li>
      </ul>
    </nav>
  );
}

export default Navbar;