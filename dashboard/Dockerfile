FROM node:20-alpine AS build

WORKDIR /app
COPY package.json ./
RUN npm install

COPY . .
ARG VITE_WS_BASE=ws://localhost:8000
ENV VITE_WS_BASE=$VITE_WS_BASE
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
