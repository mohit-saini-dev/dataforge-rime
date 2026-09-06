import asyncio
from datetime import datetime, timezone
import secrets
from typing import Any, Dict, List, Optional


class MockTravelDB:
    """In-memory thread-safe mock storage for flight bookings and hotel reservations."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.reset()

    def reset(self) -> None:
        """Reset mock database state for testing isolation."""
        self.flights: List[Dict[str, Any]] = [
            {
                "flight_id": "FL-101",
                "airline": "Skyline Express",
                "origin": "DEL",
                "destination": "BOM",
                "departure_time": "08:30 AM",
                "arrival_time": "10:45 AM",
                "price": 4500,
                "currency": "INR",
                "available_seats": 12,
            },
            {
                "flight_id": "FL-204",
                "airline": "Air Indigo",
                "origin": "DEL",
                "destination": "BOM",
                "departure_time": "02:15 PM",
                "arrival_time": "04:30 PM",
                "price": 5200,
                "currency": "INR",
                "available_seats": 5,
            },
            {
                "flight_id": "FL-309",
                "airline": "Jet Airways",
                "origin": "DEL",
                "destination": "BLR",
                "departure_time": "11:00 AM",
                "arrival_time": "01:45 PM",
                "price": 6100,
                "currency": "INR",
                "available_seats": 8,
            },
        ]
        self.hotels: List[Dict[str, Any]] = [
            {
                "hotel_id": "HT-501",
                "name": "Grand Palace Hotel",
                "city": "BOM",
                "rating": 4.5,
                "price_per_night": 3800,
                "currency": "INR",
                "available_rooms": 4,
            },
            {
                "hotel_id": "HT-502",
                "name": "Seaside Orchid Inn",
                "city": "BOM",
                "rating": 4.2,
                "price_per_night": 2900,
                "currency": "INR",
                "available_rooms": 7,
            },
        ]
        self.bookings: Dict[str, Dict[str, Any]] = {}


_db = MockTravelDB()


def _err(code: str, message: str) -> Dict[str, Any]:
    return {"status": "error", "error_code": code, "message": message}


async def search_flights(origin: str, destination: str, date: Optional[str] = None) -> Dict[str, Any]:
    """Search for available flights between two airport codes."""
    await asyncio.sleep(0.3)  # Pre-read network latency

    results = [
        {
            "flight_id": f["flight_id"],
            "airline": f["airline"],
            "origin": f["origin"],
            "destination": f["destination"],
            "departure_time": f["departure_time"],
            "arrival_time": f["arrival_time"],
            "price": {"amount": f["price"], "currency": f["currency"]},
            "status": "AVAILABLE" if f["available_seats"] > 0 else "SOLD_OUT",
        }
        for f in _db.flights
        if f["origin"].upper() == origin.strip().upper()
        and f["destination"].upper() == destination.strip().upper()
    ]
    return {
        "status": "success",
        "query": {"origin": origin.strip().upper(), "destination": destination.strip().upper(), "date": date},
        "count": len(results),
        "flights": results,
    }


async def book_flight(flight_id: str, passenger_name: str) -> Dict[str, Any]:
    """Book a flight seat atomically under lock after simulating transaction I/O."""
    clean_name = passenger_name.strip()
    clean_flight_id = flight_id.strip().upper()

    if not clean_name:
        return _err("INVALID_INPUT", "Passenger name cannot be empty.")

    # 1. Simulated transaction latency (safe cancellation window before any mutation)
    await asyncio.sleep(0.4)

    # 2. Atomic state mutation
    async with _db._lock:
        flight = next((f for f in _db.flights if f["flight_id"] == clean_flight_id), None)
        if not flight:
            return _err("FLIGHT_NOT_FOUND", f"Flight {clean_flight_id} not found.")

        if flight["available_seats"] <= 0:
            return _err("SOLD_OUT", f"Flight {clean_flight_id} is fully booked.")

        flight["available_seats"] -= 1
        booking_id = f"BK-{secrets.token_hex(4).upper()}"

        booking_record = {
            "booking_id": booking_id,
            "flight_id": clean_flight_id,
            "passenger_name": clean_name,
            "airline": flight["airline"],
            "price": {"amount": flight["price"], "currency": flight["currency"]},
            "status": "CONFIRMED",
            "booked_at": int(datetime.now(timezone.utc).timestamp() * 1000),
        }
        _db.bookings[booking_id] = booking_record

    return {
        "status": "success",
        "message": "Flight successfully booked.",
        "booking": booking_record,
    }


async def cancel_booking(booking_id: str) -> Dict[str, Any]:
    """Cancel an existing booking and restore seat inventory atomically."""
    clean_booking_id = booking_id.strip().upper()

    await asyncio.sleep(0.3)  # Cancellation latency

    async with _db._lock:
        record = _db.bookings.get(clean_booking_id)
        if not record:
            return _err("BOOKING_NOT_FOUND", f"Booking ID {clean_booking_id} not found.")

        if record["status"] == "CANCELLED":
            return _err("ALREADY_CANCELLED", f"Booking {clean_booking_id} is already cancelled.")

        record["status"] = "CANCELLED"
        flight_id = record["flight_id"]

        flight = next((f for f in _db.flights if f["flight_id"] == flight_id), None)
        if flight:
            flight["available_seats"] += 1

        refund_details = {
            "booking_id": clean_booking_id,
            "status": "CANCELLED",
            "refund": record["price"],
            "cancelled_at": int(datetime.now(timezone.utc).timestamp() * 1000),
        }

    return {
        "status": "success",
        "message": f"Booking {clean_booking_id} has been cancelled.",
        "cancellation": refund_details,
    }


async def search_hotels(city: str, nights: int = 1) -> Dict[str, Any]:
    """Search available hotels with total price calculations."""
    clean_city = city.strip().upper()
    valid_nights = max(1, nights)

    await asyncio.sleep(0.3)

    results = [
        {
            "hotel_id": h["hotel_id"],
            "name": h["name"],
            "city": h["city"],
            "rating": h["rating"],
            "price_per_night": {"amount": h["price_per_night"], "currency": h["currency"]},
            "total_price": {"amount": h["price_per_night"] * valid_nights, "currency": h["currency"]},
            "available_rooms": h["available_rooms"],
        }
        for h in _db.hotels
        if h["city"].upper() == clean_city
    ]
    return {
        "status": "success",
        "city": clean_city,
        "nights": valid_nights,
        "count": len(results),
        "hotels": results,
    }