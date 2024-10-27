import { Body, Controller, Post, ValidationPipe } from '@nestjs/common';
import { AuthService } from './auth.service';
import { CreateUserDto } from 'src/users/dto/create-user.dto';
import { LoginDto } from './dto/login.dto';

@Controller('auth')
export class AuthController {

    constructor( private authService : AuthService ){}

    @Post('/register')
    register(@Body(new ValidationPipe()) createUserDto : CreateUserDto ){
        return this.authService.register(createUserDto)
    }

    @Post('/login')
    login (@Body(new ValidationPipe()) loginDto : LoginDto ) {
        console.log('here');
        return this.authService.login(loginDto)
    }


}
