# Use the official Node.js image as a base image
FROM node:latest

# Set the working directory in the container to /app
WORKDIR /app

# Copy package.json and package-lock.json to /app
COPY package*.json ./

# Install project dependencies
RUN npm install

# Copy the rest of the project files to /app
COPY . .

# Expose port 3000 for the Nest.js application
EXPOSE 3000

# Start the application with npm start
CMD [ "npm", "run", "start:dev" ]