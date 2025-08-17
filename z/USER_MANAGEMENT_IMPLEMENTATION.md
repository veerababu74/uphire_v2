# User Management & Admin Access Feature Implementation

## Overview
This implementation adds comprehensive user management functionality to the resume search system, allowing designated admin users to access and search through all documents across all users, while regular users remain restricted to their own documents.

## Key Features Implemented

### 1. User Management System
- **User CRUD Operations**: Complete Create, Read, Update, Delete operations for users
- **Admin Status Management**: Users can be designated as admins with special privileges
- **Unique Constraints**: Enforced unique user_id and email constraints
- **MongoDB Collection**: New `users` collection with proper indexing

### 2. Admin Access Control
- **Admin Users**: Can search ALL documents across ALL users
- **Regular Users**: Can only search their own documents
- **Security**: Defaults to restricted access on errors for security

### 3. Enhanced Search APIs
All search endpoints now support the admin/regular user distinction:
- `/search/mango` - Mango retriever with admin support
- `/search/langchain` - LangChain retriever with admin support
- `/search/mango/search-by-jd` - Job description search with admin support
- `/search/langchain/search-by-jd` - Job description search with admin support
- `/rag/vector-similarity-search` - RAG vector search with admin support
- `/rag/llm-context-search` - RAG LLM search with admin support
- `/rag/search-by-jd` - RAG job description search with admin support
- `/ai/search` - Enhanced vector search with admin support

## Files Created

### 1. User Schema (`schemas/user_schemas.py`)
```python
- UserCreate: Schema for creating new users
- UserUpdate: Schema for updating existing users  
- UserResponse: Schema for user API responses
- UserSearchRequest: Enhanced search request with admin support
```

### 2. User Operations (`mangodatabase/user_operations.py`)
```python
- UserOperations class with methods:
  - create_user()
  - get_user_by_id()
  - update_user()
  - delete_user()
  - list_users()
  - is_admin_user()
  - get_admin_users()
```

### 3. User Management API (`apis/user_management.py`)
```python
- POST /users/ - Create user
- GET /users/{user_id} - Get user
- PUT /users/{user_id} - Update user
- DELETE /users/{user_id} - Delete user
- GET /users/ - List users (with pagination)
- GET /users/admin/list - List admin users
- GET /users/{user_id}/admin-status - Check admin status
```

## Files Modified

### 1. Database Client (`mangodatabase/client.py`)
- Added `get_users_collection()` function

### 2. Retriever API (`apis/retriever_api.py`)
- Added user operations initialization
- Added `get_effective_user_id_for_search()` helper function
- Updated all search endpoints to support admin users
- Updated documentation to reflect new access control

### 3. RAG Search API (`apis/rag_search.py`)
- Added user operations initialization
- Added `get_effective_user_id_for_search()` helper function
- Updated all search endpoints to support admin users
- Updated request schemas to reflect new behavior

### 4. Vector Search API (`apis/vector_search.py`)
- Added user operations initialization
- Added `get_effective_user_id_for_search()` helper function
- Updated search schema descriptions

### 5. Main Functions (`main_functions.py`)
- Added `create_users_collection_indexes()` function
- Updated `initialize_application_startup()` to include users collection

### 6. Main Application (`main.py`)
- Added user management router import and registration

### 7. Search Schemas (`schemas/vector_search_scehma.py`)
- Updated user_id field description to reflect admin behavior

## How It Works

### Admin User Workflow
1. User makes search request with their `user_id`
2. System checks if user has `is_admin: true` in users collection
3. If admin: `effective_user_id = None` (searches ALL documents)
4. Search is performed without user filtering
5. Results include documents from all users

### Regular User Workflow
1. User makes search request with their `user_id`
2. System checks if user has `is_admin: false` or doesn't exist
3. If regular user: `effective_user_id = user_id` (searches only their documents)
4. Search is performed with user_id filtering
5. Results include only their own documents

### Security Features
- **Default Restriction**: On database errors, defaults to restricting access
- **Unique Constraints**: Prevents duplicate user_ids and emails
- **Admin Logging**: All admin searches are logged for auditing
- **Input Validation**: Comprehensive validation on all user inputs

## Database Schema

### Users Collection
```javascript
{
  "_id": ObjectId,
  "user_id": "unique_user_identifier",    // Unique index
  "email": "user@example.com",           // Unique index
  "name": "User Full Name",              // Optional
  "is_admin": false,                     // Index for admin queries
  "created_at": ISODate,                 // Index for sorting
  "updated_at": ISODate
}
```

### Indexes Created
- `user_id` (unique)
- `email` (unique)
- `is_admin` 
- `created_at`
- `is_admin_1_created_at_-1` (compound)

## API Usage Examples

### Create Admin User
```bash
POST /users/
{
  "user_id": "admin001",
  "email": "admin@company.com",
  "name": "System Administrator",
  "is_admin": true
}
```

### Create Regular User
```bash
POST /users/
{
  "user_id": "user001", 
  "email": "user@company.com",
  "name": "Regular User",
  "is_admin": false
}
```

### Search as Admin (searches all documents)
```bash
POST /search/mango
{
  "user_id": "admin001",
  "query": "python developer",
  "limit": 10
}
```

### Search as Regular User (searches only their documents)
```bash
POST /search/mango
{
  "user_id": "user001", 
  "query": "python developer",
  "limit": 10
}
```

### Check Admin Status
```bash
GET /users/admin001/admin-status
# Returns: {"user_id": "admin001", "is_admin": true}
```

## Migration Notes

### For Existing Systems
1. Run the application once to create the users collection and indexes
2. Create admin users as needed using the API
3. Existing search functionality continues to work unchanged
4. Regular users will be restricted to their own documents by default

### Backward Compatibility
- All existing search APIs continue to work
- No breaking changes to existing request/response formats
- Only behavior change is the access control based on admin status

## Security Considerations

1. **Admin Privilege Escalation**: Carefully control who can create admin users
2. **Audit Logging**: Admin searches are logged for security auditing
3. **Error Handling**: Defaults to restrictive access on database errors
4. **Input Validation**: All user inputs are validated and sanitized
5. **Unique Constraints**: Prevents duplicate accounts and identity confusion

## Future Enhancements

1. **Role-Based Access**: Extend beyond admin/regular to multiple role types
2. **Department-Based Access**: Allow access to specific user groups
3. **Audit Logs**: Comprehensive audit trail for all admin actions
4. **API Rate Limiting**: Implement rate limiting for admin endpoints
5. **User Sessions**: Add session management and authentication
