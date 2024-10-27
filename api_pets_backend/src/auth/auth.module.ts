import { Module } from '@nestjs/common';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { UsersModule } from '../users/users.module';
import { JwtModule } from '@nestjs/jwt';
import { jwtConstants } from './constants';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Users } from 'src/users/entities/users.entity';
import { AuthGuard } from './auth.guards';

@Module({
  imports : [UsersModule,
    JwtModule.register(
      {
        global : true,
        secret : jwtConstants.secret,
        signOptions : { expiresIn : '9999s' }
      }
    ), 
    TypeOrmModule.forFeature([Users])

  ],
  controllers: [AuthController],
  providers: [AuthService, AuthGuard],
  exports : [AuthGuard]
})
export class AuthModule {}
