FROM node:20-alpine AS build

WORKDIR /app
COPY package.json ./
RUN npm install

COPY . .
ARG VITE_WS_BASE=ws://localhost:8010
ARG VITE_DEFAULT_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ARG VITE_API_HTTP_TARGET=http://127.0.0.1:8010
ENV VITE_WS_BASE=$VITE_WS_BASE
ENV VITE_DEFAULT_MODEL=$VITE_DEFAULT_MODEL
ENV VITE_API_HTTP_TARGET=$VITE_API_HTTP_TARGET
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
