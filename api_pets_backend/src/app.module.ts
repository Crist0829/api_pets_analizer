import { MiddlewareConsumer, Module, NestModule, RequestMethod } from '@nestjs/common';
import { logger } from './pets/logger.mileware';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import typeorm from './config/typeorm';
import { PetsModule } from './pets/pets.module';


@Module({
  imports: [ConfigModule.forRoot(
    {
      isGlobal: true,
      load: [typeorm]
    }
  ), 
  TypeOrmModule.forRootAsync({
    inject: [ConfigService],
    useFactory: async (configService: ConfigService) => (configService.get('typeorm'))
  }),  
  AuthModule, UsersModule, PetsModule],
})
export class AppModule implements NestModule{
  configure(consumer: MiddlewareConsumer) {
    consumer
    .apply(logger)
    .forRoutes({path : 'pets', method : RequestMethod.GET})
  }
}
