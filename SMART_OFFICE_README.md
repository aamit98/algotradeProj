# Smart-Office Asset Manager

A containerized asset management system for booking office resources: desks, meeting rooms, laptops, and more.

**Stack:** .NET 9 | React + TypeScript | MobX | MUI | Docker

---

## Run the Application

### Prerequisites

Docker Desktop installed and running

### Start Everything

```bash
docker compose up --build
```

Wait for all services to initialize (around 30 to 60 seconds on first run), then open:

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Auth Service Swagger | http://localhost:5001/swagger |
| Resource Service Swagger | http://localhost:5002/swagger |

---

## Swagger API Testing

Both backend services expose interactive Swagger documentation with JWT Bearer authentication:

### Auth Service (http://localhost:5001/swagger)

- Test registration and login endpoints
- Copy the returned JWT token for Resource Service testing

### Resource Service (http://localhost:5002/swagger)

- Click "Authorize" button at the top
- Paste your JWT token (format: `Bearer <your_token>`)
- Test all asset endpoints with authentication

---

## Test the Role Based Access

1. Open http://localhost:5173
2. Click Register and create a user with **Admin** role
3. Open an incognito window and register a **Member** user
4. Admin sees the "Add New Asset" button
5. Member does not see it

---

## Architecture

Two isolated microservices sharing only a JWT contract:

```
┌──────────────────┐
│     Frontend     │
│  React + MobX    │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌─────────────┐
│  Auth  │  │  Resource   │
│Service │  │  Service    │
└───┬────┘  └──────┬──────┘
    │              │
    ▼              ▼
┌────────┐  ┌──────────┐
│Postgres│  │ MongoDB  │
└────────┘  └──────────┘
```

### Why different databases?

User credentials need transactional integrity (PostgreSQL). Assets and bookings are document oriented data that benefits from MongoDB's flexible schema and atomic update operators. This separation also demonstrates proper data isolation between services.

---

## API Reference

### Auth Service (Port 5001)

Swagger UI: http://localhost:5001/swagger

| Endpoint | Method | Body | Returns |
|----------|--------|------|---------|
| `/api/auth/register` | POST | `{ username, password, fullName, role }` | `{ token }` |
| `/api/auth/login` | POST | `{ username, password }` | `{ token }` |

JWT contains: `sub` (userId), `role` (Admin or Member), `FullName`

### Resource Service (Port 5002)

Swagger UI: http://localhost:5002/swagger

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/assets` | GET | Any | Paginated list with optional `?isAvailable=true` filter |
| `/api/assets/{id}` | GET | Any | Single asset details |
| `/api/assets/stats` | GET | Any | Returns `{ total, available, inUse }` |
| `/api/assets` | POST | Admin | Create new asset |
| `/api/assets/{id}` | PUT | Any | Book or release asset |
| `/api/assets/{id}` | DELETE | Admin | Delete asset |

---

## Technical Decisions

### Atomic Booking (Preventing Race Conditions)

**Problem:** Two users click "Book" on the same desk simultaneously. A naive read then write could let both succeed.

**Solution:** MongoDB conditional update. Booking succeeds only if `IsAvailable == true` at write time:

```csharp
var filter = Builders<Asset>.Filter.And(
    Builders<Asset>.Filter.Eq(a => a.Id, id),
    Builders<Asset>.Filter.Eq(a => a.IsAvailable, true)
);
var result = await _collection.UpdateOneAsync(filter, update);

if (result.ModifiedCount == 0)
    return null; // 409 Conflict, someone else got it
```

No locks needed. The database guarantees atomicity.

### Server Side Pagination

Initially loaded all assets to memory. Fine for 20 items, problematic for 10,000. Fixed by pushing pagination to MongoDB:

```csharp
await _collection.Find(filter)
    .Skip((page - 1) * pageSize)
    .Limit(pageSize)
    .ToListAsync();
```

Memory usage stays constant regardless of dataset size.

### JWT Secret Consistency

Both services share the same `JWT_SECRET` via environment variables. Auth Service signs tokens; Resource Service validates them. No inter service calls required.

### Role Selection at Registration

Exposed for easy RBAC testing. In production, admin assignment would be handled via seed scripts or internal tooling.

---

## Running Tests

```bash
# Backend
cd backend/auth-service && dotnet test
cd backend/resource-service && dotnet test

# Frontend
cd frontend && npm test
```

---

## Extra Features

| Feature | Purpose |
|---------|---------|
| Server side pagination | Stable response times at scale |
| Clickable dashboard stats | Quick filtering by Total, Available, or In Use |
| Full name on bookings | Shows who booked the asset, not just an ID |
| Admin pre booking | Create assets already assigned to guests |
| Database seeding | Demo data available on first run |
| Swagger with JWT auth | Test APIs directly in browser |

---

## Project Structure

```
smart-office/
├── backend/
│   ├── auth-service/         # .NET 9 + PostgreSQL
│   └── resource-service/     # .NET 9 + MongoDB
├── frontend/                 # React + MobX + MUI
├── docker-compose.yml
└── README.md
```

---

## Stack

| Layer | Technology |
|-------|------------|
| Auth Backend | .NET 9, EF Core, PostgreSQL, BCrypt |
| Resource Backend | .NET 9, MongoDB.Driver |
| Frontend | React 19, TypeScript, MobX, MUI 7, Vite |
| Infrastructure | Docker Compose, PostgreSQL 15, MongoDB 7 |

---

## Tooling Disclosure

I used the following tools during development:

- **ChatGPT** | Brainstormed atomic booking approaches, validated JWT flow design
- **GitHub Copilot** | Autocomplete for MUI boilerplate and test setup
- **MongoDB Compass** | Document inspection during debugging
- **DBeaver** | Verified PostgreSQL migrations

I take full responsibility for this codebase. I understand the JWT validation flow between services, Docker networking, atomic update patterns, and can explain any implementation detail in depth.
