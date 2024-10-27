import { Controller, Get, Post, Body, Patch, Param, Delete, ValidationPipe, UseGuards, UseInterceptors, UploadedFile } from '@nestjs/common';
import { PetsService } from './pets.service';
import { CreatePetDto } from './dto/create-pet.dto';
import { UpdatePetDto } from './dto/update-pet.dto';
import { User } from 'src/common/user.decorator';
import { Users } from 'src/users/entities/users.entity';
import { AuthGuard } from 'src/auth/auth.guards';
import { FileInterceptor } from '@nestjs/platform-express';
import { uploadInterceptor } from './uploadInterceptor';


@Controller('pets')
export class PetsController {
  
  constructor(private readonly petsService: PetsService) {
    
  }

  @Post('/')
  @UseGuards(AuthGuard)
  create(@Body(new ValidationPipe()) createPetDto: CreatePetDto, @User() user : Users ) {
    console.log(user)
    return this.petsService.create(createPetDto, user);
  }

  @Get('/me' )
  @UseGuards(AuthGuard)
  findAll(@User() user : Users) {
    return this.petsService.findByUser(user.id) 
  }

  @Get("/all")
  @UseGuards(AuthGuard)
  findAllPets() {
    return this.petsService.findAll();
  }

  @Get(':id')
  @UseGuards(AuthGuard)
  findOne(@Param('id') id: string) {
    return this.petsService.findOne(+id);
  }

  @Patch(':id')
  @UseGuards(AuthGuard)
  update(@Param('id') id: string, @Body() updatePetDto: UpdatePetDto) {
    return this.petsService.update(+id, updatePetDto);
  }

  @Delete(':id')
  remove(@Param('id') id: string) {
    return this.petsService.remove(+id);
  }


  @Post('/upload')
  @UseGuards(AuthGuard)
  @UseInterceptors(uploadInterceptor)
  uploadPetImage(@UploadedFile() file: Express.Multer.File, @User() user : Users){
    return this.petsService.uploadImage(user.id, file, null)
  }


}
