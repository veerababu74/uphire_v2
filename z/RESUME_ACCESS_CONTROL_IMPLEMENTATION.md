# Resume API User Access Control Implementation

## Overview
Implemented user-based access control for all resume CRUD operations, similar to the manual search functionality. Users are now categorized into two groups with different permissions:

## User Permission Levels

### 1. **Admin Users** (Users in `users_collection`)
- ✅ Can perform **all operations** on **any resume**
- ✅ Can create resumes for any user_id
- ✅ Can read, update, delete any resume
- ✅ Can list all resumes in the system
- ✅ Can update embeddings for all resumes

### 2. **Regular Users** (Users NOT in `users_collection`)
- ✅ Can only perform operations on **their own resumes** (matching their `user_id`)
- ✅ Can create resumes only for their own user_id
- ✅ Can read, update, delete only their own resumes
- ✅ Can list only their own resumes
- ✅ Can update embeddings only for their own resumes

## API Changes

### Required Parameter Added
All endpoints now require:
- `requesting_user_id` (Query parameter): The user_id of the user making the request

### Modified Endpoints

#### 1. **POST /resumes/** - Create Resume
```http
POST /resumes/?requesting_user_id=user123
Content-Type: application/json

{
  "user_id": "user123",
  "username": "John Doe",
  // ... other resume fields
}
```

**Access Control:**
- Admin users: Can create resumes for any user_id
- Regular users: Can only create resumes with their own user_id

#### 2. **PUT /resumes/{resume_id}** - Update Resume
```http
PUT /resumes/648f1234567890abcdef?requesting_user_id=user123
Content-Type: application/json

{
  "user_id": "user123",
  "username": "John Doe Updated",
  // ... other resume fields
}
```

**Access Control:**
- Admin users: Can update any resume
- Regular users: Can only update their own resumes

#### 3. **GET /resumes/{resume_id}** - Get Resume
```http
GET /resumes/648f1234567890abcdef?requesting_user_id=user123
```

**Access Control:**
- Admin users: Can view any resume
- Regular users: Can only view their own resumes

#### 4. **DELETE /resumes/{resume_id}** - Delete Resume
```http
DELETE /resumes/648f1234567890abcdef?requesting_user_id=user123
```

**Access Control:**
- Admin users: Can delete any resume
- Regular users: Can only delete their own resumes

#### 5. **GET /resumes/** - List Resumes
```http
GET /resumes/?requesting_user_id=user123&skip=0&limit=10
```

**Access Control:**
- Admin users: Returns all resumes in the system
- Regular users: Returns only their own resumes

#### 6. **POST /resumes/update-embeddings** - Update All Embeddings
```http
POST /resumes/update-embeddings?requesting_user_id=user123
```

**Access Control:**
- Admin users: Updates embeddings for all resumes
- Regular users: Updates embeddings only for their own resumes

#### 7. **PATCH /resumes/{resume_id}/regenerate-vectors** - Regenerate Vectors
```http
PATCH /resumes/648f1234567890abcdef/regenerate-vectors?requesting_user_id=user123
```

**Access Control:**
- Admin users: Can regenerate vectors for any resume
- Regular users: Can only regenerate vectors for their own resumes

#### 8. **PATCH /resumes/{resume_id}** - Partial Update
```http
PATCH /resumes/648f1234567890abcdef?requesting_user_id=user123&regenerate_vectors=true
Content-Type: application/json

{
  "skills": ["python", "javascript", "react"]
}
```

**Access Control:**
- Admin users: Can partially update any resume
- Regular users: Can only partially update their own resumes

## Implementation Details

### Core Functions Added

#### 1. `get_effective_user_id_for_resume_operations(requesting_user_id)`
- Checks if user exists in users collection
- Returns `None` for admin users (access all)
- Returns `requesting_user_id` for regular users (access own only)

#### 2. `validate_user_access_to_resume(requesting_user_id, resume)`
- Validates if user has access to specific resume
- Admin users: Always returns `True`
- Regular users: Returns `True` only if resume belongs to them

### New Methods in ResumeOperations

#### 1. `list_resumes_by_user(user_id, skip, limit)`
- Lists resumes for a specific user with pagination
- Used when regular users request resume listing

#### 2. `update_vector_embeddings_by_user(user_id)`
- Updates vector embeddings only for resumes belonging to specific user
- Used when regular users request embedding updates

## Security Features

### 1. **Permission Validation**
- Every operation validates user permissions before execution
- Users cannot access resumes they don't own (unless admin)

### 2. **User ID Enforcement**
- Regular users cannot create/update resumes with different user_ids
- Admin users can work with any user_id

### 3. **Error Handling**
- `403 Forbidden` for permission violations
- `404 Not Found` for non-existent resumes
- `400 Bad Request` for validation errors

### 4. **Audit Logging**
- All operations log which user performed what action
- Tracks access attempts and permission checks

## HTTP Status Codes

- `200 OK` - Successful operation
- `201 Created` - Resume created successfully
- `400 Bad Request` - Invalid input or missing required fields
- `403 Forbidden` - Permission denied
- `404 Not Found` - Resume not found
- `500 Internal Server Error` - Server error

## Usage Examples

### Admin User Operations
```python
# Admin can access any resume
admin_user_id = "admin123"  # exists in users_collection

# Create resume for any user
POST /resumes/?requesting_user_id=admin123
{
  "user_id": "any_user_id",
  "username": "Any User"
}

# Update any resume
PUT /resumes/resume_id?requesting_user_id=admin123

# List all resumes
GET /resumes/?requesting_user_id=admin123
```

### Regular User Operations
```python
# Regular user can only access their own resumes
regular_user_id = "user123"  # NOT in users_collection

# Can only create resume for themselves
POST /resumes/?requesting_user_id=user123
{
  "user_id": "user123",  # Must match requesting_user_id
  "username": "John Doe"
}

# Can only update their own resumes
PUT /resumes/their_resume_id?requesting_user_id=user123

# Lists only their own resumes
GET /resumes/?requesting_user_id=user123
```

## Migration Impact

- **Non-breaking for admin users**: Users in users_collection continue to have full access
- **Enforcement for regular users**: Users not in users_collection are now restricted to their own data
- **API parameter addition**: All endpoints now require `requesting_user_id` parameter
- **Consistent with manual search**: Uses same access control logic as the manual search API

This implementation ensures data security while maintaining administrative flexibility, following the same pattern established in the manual search functionality.
