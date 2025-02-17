# ✈️ [SKY - NAV](https://skynav.pythonanywhere.com/)

## 🌍 Navigation System for Small Aircrafts, Helicopters, and Drones

SkyNav-AirArmy is a **real-time navigation system** designed to guide **small aircrafts, helicopters, and drones** from **Point A to Point B** while avoiding **hazardous weather conditions** such as snow, hail, and storms. It utilizes the **A* (A-star) pathfinding algorithm** to determine the **shortest and safest route** for flying.

---

## 🚀 Features
✔ **Shortest Path Calculation** - Uses **A* Algorithm** to find the **most efficient** route.  
✔ **Weather-Aware Routing** - Avoids areas with **bad weather (snow, hail, storms, etc.)**.  
✔ **User Input for Coordinates** - Enter **source and destination coordinates** to get an optimized path.  
✔ **Real-time Updates** - Dynamically adjusts paths based on **weather conditions**.  
✔ **Scalability** - Can be expanded to integrate **live weather APIs** for real-time data.  

---

## 🛠️ Technologies Used
| **Technology** | **Purpose** |
|--------------|------------|
| **Python** | Backend logic, A* algorithm |
| **JavaScript (Node.js)** | CLI-based distance calculation |
| **Flask (or Express.js)** | API for distance & weather handling |
| **OpenWeather API (Future)** | Fetch real-time weather data |

---

## 📥 Installation & Setup

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/BetterThanYou73/SkyNav-AirArmy.git
cd SkyNav-AirArmy
```
### **2️⃣ Install Dependencies**
Depending on which part of the project you are running, install the necessary dependencies.
### 2️⃣ Install Dependencies
Depending on which part of the project you are running, install the necessary dependencies.

#### 🐍 For Python Backend
```sh
pip install flask
```
