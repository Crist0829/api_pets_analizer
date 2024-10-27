import { Module } from '@nestjs/common';
import { PetsService } from './pets.service';
import { PetsController } from './pets.controller';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Users } from 'src/users/entities/users.entity';
import { Pets } from './entities/pet.entity';

@Module({
  imports : [TypeOrmModule.forFeature([Users, Pets])],
  controllers: [PetsController],
  providers: [PetsService],
})
export class PetsModule {}
