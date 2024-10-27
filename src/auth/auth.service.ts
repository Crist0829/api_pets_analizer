import { Injectable } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { InjectRepository } from '@nestjs/typeorm';
import { CreateUserDto } from 'src/users/dto/create-user.dto';
import { Users } from 'src/users/entities/users.entity';
import { Repository } from 'typeorm';
import * as bcrypt from 'bcrypt';
import { LoginDto } from './dto/login.dto';

@Injectable()
export class AuthService {
  constructor(
    @InjectRepository(Users)
    private userRepository: Repository<Users>,
    private jwtService: JwtService
  ) {}

  async login(loginDto : LoginDto): Promise<{access_token : string } | { error : string }> {

    let user = await this.userRepository.findOne({ where: { email: loginDto.email } });

    if (!user) {
      return {
        error : 'No hay un usuario con ese email registrado'
      }
    }

    if(!bcrypt.compare(loginDto.password, user.password)){
        return {
            error : 'Contraseña incorrecta'
        }
    }

    const payload = { id: user.id, name : user.name };

    return {
        access_token : await this.jwtService.signAsync(payload)
    }
    
  }


  async register( createUserDto : CreateUserDto ) : Promise<{access_token : string} | { error : string}> {

    let user = await this.userRepository.findOne({ where: { email: createUserDto.email } });

    if(user){
        return {
            error : 'Ya existe un usuario con ese email'
        }
    }

    const userData = {
        name : createUserDto.name, 
        email : createUserDto.email, 
        password : await bcrypt.hash(createUserDto.password, 10)
    }

    user =  await this.userRepository.save(userData)
    const payload = { id: user.id, name : user.name };

    return {
        access_token : await this.jwtService.signAsync(payload)
    }


  }

}
