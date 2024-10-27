import { IsString, IsNotEmpty, IsIn, Min, Max, IsEnum} from 'class-validator';
import { PetsType } from '../entities/enums';

export class CreatePetDto {

    @IsNotEmpty()
    name : string

    @IsNotEmpty()
    color : string

    @IsEnum(PetsType, { message: 'type must be one of: cat, dog, other' })
    type: PetsType;

    @Min(1)
    height : number


}
