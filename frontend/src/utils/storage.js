// Generate a random user ID
export const generateUserId = () => {
  return "u" + Math.random().toString(36).substr(2, 9);
};

// Generate a random conversation ID
export const generateConversationId = () => {
  return "c" + Math.random().toString(36).substr(2, 9);
};

// Local storage keys
const STORAGE_KEYS = {
  USER_ID: "ai_agents_user_id",
};

// Get user ID from local storage or generate a new one
export const getUserId = () => {
  let userId = localStorage.getItem(STORAGE_KEYS.USER_ID);
  if (!userId) {
    userId = generateUserId();
    localStorage.setItem(STORAGE_KEYS.USER_ID, userId);
  }
  return userId;
};

// Clear user ID from local storage
export const clearUserId = () => {
  localStorage.removeItem(STORAGE_KEYS.USER_ID);
};

// Check if user ID exists in local storage
export const hasUserId = () => {
  return localStorage.getItem(STORAGE_KEYS.USER_ID) !== null;
};
