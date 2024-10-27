import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { MicroserviceOptions, Transport } from '@nestjs/microservices';
import { join } from 'path';
import * as express from 'express';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  app.use('/uploads', express.static(join(__dirname, '../..', 'uploads')));

  app.connectMicroservice<MicroserviceOptions> ({
    transport: Transport.RMQ,
    options: {
      urls: ['amqp://user:password@localhost:5672'], // Cambia según tus credenciales
      queue: 'image.uploaded',
      queueOptions: { durable: true },
    },
  });

  //app.useGlobalInterceptors(new ResponseInterceptor());
  await app.listen(3000);
}
bootstrap();
